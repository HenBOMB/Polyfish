#!/usr/bin/env bash
# Snapshot the experiment record to a second disk / mounted volume.
# Safe to run while training is running: live files are only ever read, and the
# snapshot is staged then renamed into place, so a reader never sees a partial
# directory. Non-zero exit means the snapshot is missing or suspect.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: backup_experiment_record.sh [DEST]

DEST, or $POLYFISH_BACKUP_DIR, is a directory on another disk/volume.
Creates DEST/<UTC timestamp>/ holding the record, a MANIFEST and SHA256SUMS,
and writes the snapshot name to DEST/LATEST.

Env:
  POLYFISH_BACKUP_DIR     destination (when DEST is not given)
  POLYFISH_RUN_DIR        source dir (default: the polyfish-rs dir above this script)
  POLYFISH_BACKUP_EXTRA   extra space-separated paths, relative to the source dir
  POLYFISH_BACKUP_KEEP    keep only the newest N snapshots (default: keep all)
  POLYFISH_BACKUP_LINK    0 disables hardlink reuse of unchanged checkpoints
  POLYFISH_BACKUP_REMOTE  optional mirror: s3://..., gs://... or host:/path

Exit: 0 ok, 1 the snapshot failed, 2 bad usage, 3 published but an item is
suspect (see MANIFEST).
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

DEST="${1:-${POLYFISH_BACKUP_DIR:-}}"
if [ -z "$DEST" ]; then
    usage >&2
    exit 2
fi
if [ "$#" -gt 1 ]; then
    echo "unexpected extra arguments: ${*:2}" >&2
    exit 2
fi

SRC="${POLYFISH_RUN_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [ ! -d "$SRC" ]; then
    echo "source dir not found: $SRC" >&2
    exit 1
fi

ITEMS="training_log.csv ladder.json moves_by_turn.json .anchor_state.json .anchor_decay_start checkpoints ${POLYFISH_BACKUP_EXTRA:-}"

if command -v sha256sum >/dev/null 2>&1; then
    SUM_CMD=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
    SUM_CMD=(shasum -a 256)
else
    echo "no sha256sum or shasum on PATH; refusing to write an unverifiable snapshot" >&2
    exit 1
fi

if stat -c '%s %y' "$SRC" >/dev/null 2>&1; then
    sig() { stat -c '%s %y' "$1"; }
else
    sig() { stat -f '%z %m' "$1"; }
fi
fsize() { wc -c < "$1" | tr -d ' '; }

if ! mkdir -p "$DEST"; then
    echo "cannot create destination: $DEST" >&2
    exit 1
fi
DEST="$(cd "$DEST" && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$DEST/$STAMP"
SEQ=2
while [ -e "$OUT" ]; do
    if [ "$SEQ" -gt 99 ]; then
        echo "too many snapshots in one second: $OUT" >&2
        exit 1
    fi
    OUT="$DEST/$STAMP-$SEQ"
    SEQ=$((SEQ + 1))
done
STAMP="${OUT##*/}"
STAGE="$DEST/.incoming-$STAMP.$$"

WORK="$(mktemp -d)"
ITEM_LOG="$WORK/items"
: > "$ITEM_LOG"
cleanup() {
    rm -rf "$WORK" 2>/dev/null || true
    rm -rf "$STAGE" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$STAGE"

PREV=""
if [ -f "$DEST/LATEST" ]; then
    PREV_NAME="$(cat "$DEST/LATEST")"
    if [ -n "$PREV_NAME" ] && [ -d "$DEST/$PREV_NAME" ]; then
        PREV="$DEST/$PREV_NAME"
    fi
fi

SUSPECT=0

# Copy, retrying while the source keeps changing under us (training is live).
# The size comparison is what catches a truncate-and-rewrite: mtime alone can
# be identical on both sides of one, since the rewrite restores the old size.
copy_stable() {
    local src=$1 dst=$2 attempt=0 before after
    while :; do
        before="$(sig "$src")"
        cp -p "$src" "$dst" || return 1
        after="$(sig "$src")"
        if [ "$before" = "$after" ] && [ "$(fsize "$dst")" = "${after%% *}" ]; then
            echo stable
            return 0
        fi
        attempt=$((attempt + 1))
        if [ "$attempt" -ge 5 ]; then
            echo unstable
            return 0
        fi
        # Back off: retries with no gap all land inside the same write window.
        sleep "$attempt"
    done
}

# training_log.csv is rewritten in place (not renamed), so a copy can land
# mid-write; drop a torn final line rather than keep a corrupt row.
check_csv() {
    local f=$1 note="" bad
    if [ -n "$(tail -c 1 "$f")" ]; then
        sed '$d' "$f" > "$f.trim" && mv "$f.trim" "$f"
        note="trimmed_torn_line"
    fi
    bad="$(awk -F, 'NR==1{n=NF; next} NF!=n{c++} END{print c+0}' "$f")"
    if [ "$bad" != "0" ]; then
        note="${note:+$note,}field_count_mismatch=$bad"
    fi
    echo "${note:-ok}"
}

check_json() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo unchecked
        return 0
    fi
    if python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$1" >/dev/null 2>&1; then
        echo ok
    else
        echo invalid_json
    fi
}

copy_file() {
    local rel=$1 src=$2 dst=$3 state note attempt=0
    mkdir -p "$(dirname "$dst")"
    while :; do
        state="$(copy_stable "$src" "$dst")" || return 1
        case "$rel" in
            *.csv) note="$(check_csv "$dst")" ;;
            *.json) note="$(check_json "$dst")" ;;
            *) note=ok ;;
        esac
        [ "$note" != invalid_json ] && break
        attempt=$((attempt + 1))
        [ "$attempt" -ge 3 ] && break
    done
    if [ "$state" = unstable ]; then
        note="${note},source_still_changing"
    fi
    case "$note" in *invalid_json*|*source_still_changing*) SUSPECT=1 ;; esac
    printf 'item\t%s\tfile\t%s\t%s\n' "$rel" "$(fsize "$dst")" "$note" >> "$ITEM_LOG"
}

# Checkpoints are immutable once written, so an identically sized file from the
# previous snapshot is hardlinked instead of re-copied.
copy_dir() {
    local rel=$1 src=$2 dst=$3 f rf state bytes=0 count=0 linked=0 unstable=0
    mkdir -p "$dst"
    find "$src" -type f > "$WORK/filelist"
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        rf="${f#"$src"/}"
        mkdir -p "$dst/$(dirname "$rf")"
        if [ "${POLYFISH_BACKUP_LINK:-1}" != 0 ] && [ -n "$PREV" ] && [ -f "$PREV/$rel/$rf" ] &&
           [ "$(fsize "$PREV/$rel/$rf")" = "$(fsize "$f")" ] &&
           ln "$PREV/$rel/$rf" "$dst/$rf" 2>/dev/null; then
            linked=$((linked + 1))
        else
            state="$(copy_stable "$f" "$dst/$rf")" || return 1
            [ "$state" = unstable ] && unstable=$((unstable + 1))
        fi
        bytes=$((bytes + $(fsize "$dst/$rf")))
        count=$((count + 1))
    done < "$WORK/filelist"
    [ "$unstable" -gt 0 ] && SUSPECT=1
    printf 'item\t%s\tdir\t%s\tfiles=%s,hardlinked=%s,unstable=%s\n' \
        "$rel" "$bytes" "$count" "$linked" "$unstable" >> "$ITEM_LOG"
}

FOUND=0
for rel in $ITEMS; do
    src="$SRC/$rel"
    if [ -f "$src" ]; then
        FOUND=$((FOUND + 1))
        copy_file "$rel" "$src" "$STAGE/$rel" || { echo "copy failed: $rel" >&2; exit 1; }
    elif [ -d "$src" ]; then
        FOUND=$((FOUND + 1))
        copy_dir "$rel" "$src" "$STAGE/$rel" || { echo "copy failed: $rel" >&2; exit 1; }
    else
        printf 'item\t%s\tmissing\t0\tnot_present\n' "$rel" >> "$ITEM_LOG"
    fi
done

if [ "$FOUND" -eq 0 ]; then
    echo "nothing to back up: no record files under $SRC" >&2
    exit 1
fi

if ! ( cd "$STAGE" && find . -type f -exec "${SUM_CMD[@]}" {} + | LC_ALL=C sort -k2 ) > "$WORK/SHA256SUMS"; then
    echo "checksumming failed" >&2
    exit 1
fi
cp "$WORK/SHA256SUMS" "$STAGE/SHA256SUMS"
if ! ( cd "$STAGE" && "${SUM_CMD[@]}" -c SHA256SUMS >/dev/null ); then
    echo "checksum verification failed in $STAGE" >&2
    exit 1
fi

TOTAL_FILES="$(wc -l < "$STAGE/SHA256SUMS" | tr -d ' ')"
TOTAL_BYTES="$(find "$STAGE" -type f -exec wc -c {} + | awk '$2 != "total" { s += $1 } END { print s+0 }')"
GIT_COMMIT="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=no
if [ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ]; then
    GIT_DIRTY=yes
fi
STATUS=complete
if [ "$SUSPECT" -eq 1 ]; then
    STATUS=complete_with_suspect_items
fi

{
    echo "schema=1"
    echo "snapshot=$STAMP"
    echo "created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname 2>/dev/null || echo unknown)"
    echo "source=$SRC"
    echo "git_commit=$GIT_COMMIT"
    echo "git_dirty=$GIT_DIRTY"
    echo "previous=${PREV:-none}"
    cat "$ITEM_LOG"
    echo "files_total=$TOTAL_FILES"
    echo "bytes_total=$TOTAL_BYTES"
    echo "sha256sums_sha256=$( (cd "$STAGE" && "${SUM_CMD[@]}" SHA256SUMS) | cut -d' ' -f1)"
    echo "status=$STATUS"
} > "$WORK/MANIFEST"
cp "$WORK/MANIFEST" "$STAGE/MANIFEST"

if ! mv "$STAGE" "$OUT"; then
    echo "publish failed: $STAGE -> $OUT" >&2
    exit 1
fi
printf '%s\n' "$STAMP" > "$DEST/.LATEST.$$"
mv "$DEST/.LATEST.$$" "$DEST/LATEST"
echo "Snapshot $OUT ($TOTAL_FILES files, $TOTAL_BYTES bytes, status=$STATUS)"

if [ -n "${POLYFISH_BACKUP_REMOTE:-}" ]; then
    R="${POLYFISH_BACKUP_REMOTE%/}"
    case "$R" in
        s3://*)
            command -v aws >/dev/null || { echo "aws CLI not installed" >&2; exit 1; }
            aws s3 sync "$OUT" "$R/$STAMP" ;;
        gs://*)
            command -v gsutil >/dev/null || { echo "gsutil not installed" >&2; exit 1; }
            gsutil -m rsync -r "$OUT" "$R/$STAMP" ;;
        *:*)
            if command -v rsync >/dev/null; then
                rsync -a "$OUT" "$R/"
            elif command -v scp >/dev/null; then
                scp -r "$OUT" "$R/"
            else
                echo "neither rsync nor scp available" >&2
                exit 1
            fi ;;
        *)
            echo "unsupported POLYFISH_BACKUP_REMOTE: $R" >&2
            exit 1 ;;
    esac
    echo "Mirrored to $R/$STAMP"
fi

KEEP="${POLYFISH_BACKUP_KEEP:-0}"
if [ "$KEEP" -gt 0 ]; then
    find "$DEST" -mindepth 1 -maxdepth 1 -type d -name '20*T*Z*' | LC_ALL=C sort > "$WORK/snaps"
    TOTAL_SNAPS="$(wc -l < "$WORK/snaps" | tr -d ' ')"
    DROP=$((TOTAL_SNAPS - KEEP))
    if [ "$DROP" -gt 0 ]; then
        head -n "$DROP" "$WORK/snaps" | while IFS= read -r old; do
            [ -n "$old" ] && rm -rf "$old"
        done
    fi
fi

if [ "$SUSPECT" -eq 1 ]; then
    echo "one or more items are suspect; see $OUT/MANIFEST" >&2
    exit 3
fi
exit 0
