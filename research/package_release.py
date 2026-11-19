"""Build the asset-cleared research source archive from a committed revision."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    "CITATION.cff",
    "LICENSE",
    "polyfish-rs/Cargo.lock",
    "polyfish-rs/Cargo.toml",
    "polyfish-rs/LICENSE",
    "polyfish-rs/init_model.py",
    "polyfish-rs/train.py",
    "polyfish-rs/requirements.txt",
    "polyfish-rs/examples/metal_bench.rs",
    "polyfish-rs/examples/metal_parity.rs",
    "polyfish-rs/examples/tch_parity.rs",
    "polyfish-rs/examples/research_bench.rs",
    "polyfish-rs/src",
    "polyfish-rs/tests/research_simulator.rs",
    "polyfish-rs/tests/unit_push.rs",
    "polyfish-ui/LICENSE",
    "research",
    "polyfish-ui/public/research",
)
MIDGAME_EXAMPLE = "polyfish-rs/examples/midgame_bench.rs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Archive already exists: {output}")
    commit = subprocess.check_output(["git", "rev-parse", "--verify", args.ref + "^{commit}"],
                                     cwd=ROOT, text=True).strip()
    optional = subprocess.check_output(
        ["git", "ls-tree", "--name-only", commit, "--", MIDGAME_EXAMPLE], cwd=ROOT, text=True,
    ).splitlines()
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "source.zip"
        subprocess.run([
            "git", "archive", "--format=zip", "--prefix=polyfish-research-v1-rc.1/",
            "--output", str(staged), commit, "--", *INCLUDE, *optional,
        ], cwd=ROOT, check=True)
        with ZipFile(staged, "a") as archive:
            archive.writestr("polyfish-research-v1-rc.1/SOURCE_COMMIT", commit + "\n")
        try:
            os.link(staged, output)
        except FileExistsError:
            parser.error(f"Archive already exists: {output}")


if __name__ == "__main__":
    main()
