"""Compare seeded matches recorded by pre- and post-compaction Rust builds."""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


MAP_TYPES = ("drylands", "lakes", "continents", "pangea", "archipelago", "water-world")


def run(binary, tiles, topology, seed, presets, replay):
    output = subprocess.check_output([
        str(binary), "--map-size", str(tiles), "--map-type", topology,
        "--seed", str(seed), "--preset1", presets[0], "--preset2", presets[1],
        "--max-turns", "14", "--replay", str(replay),
    ], text=True)
    data = json.loads(replay.read_text())
    for tile in data["initialState"]["tiles"].values():
        for field in ("explorers", "effects"):
            if field in tile:
                tile[field].sort()
    return json.loads(output), data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    binaries = [path.resolve(strict=True) for path in (args.baseline, args.candidate)]
    cases = [(tiles, topology, seed, ("normal", "easy"))
             for tiles, topology, seed in itertools.product((121, 256, 900), MAP_TYPES, (3, 42, 971))]
    cases.extend((tiles, topology, 42, ("expert", "normal"))
                 for tiles, topology in itertools.product((121, 900), ("drylands", "archipelago", "water-world")))
    actions = 0
    with TemporaryDirectory(prefix="polyfish-board-parity-") as directory:
        for tiles, topology, seed, presets in cases:
            a, b = (run(binary, tiles, topology, seed, presets, Path(directory) / f"{i}.json")
                    for i, binary in enumerate(binaries))
            if a != b:
                different = [key for key in a[1] if a[1][key] != b[1].get(key)]
                raise RuntimeError(f"Replay diverged on {(tiles, topology, seed, presets)} in {different}")
            actions += a[0]["actions"]
    print(json.dumps({"matches": len(cases), "actions": actions, "identical": True}))


if __name__ == "__main__":
    main()
