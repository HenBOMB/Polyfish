import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help='input file path')
args = parser.parse_args()
filepath: str = args.f or "replays"

def sanitize(filepath: str):
    changed = False

    print(f"- {filepath.split('/')[-1]}")

    replay_urls = open(filepath, "r").read().split('\n');
    count = len(replay_urls)
    replay_urls = [url for url in replay_urls if "/g/" in url and "/l/" not in url]

    if count - len(replay_urls) != 0:
        changed = True
        print(f"Removed {count - len(replay_urls)} invalid replays. {len(replay_urls)} remain")

    count = len(replay_urls)
    replay_urls = list(set(replay_urls))

    if count - len(replay_urls) != 0:
        changed = True
        print(f"Removed {count - len(replay_urls)} duplicates. {len(replay_urls)} remain")

    if not changed:
        print("No changes were made")

    else:
        open(filepath, "w").write("\n".join(replay_urls))

if '.' not in filepath:
    replay_files = [f for f in os.listdir('src/scraper/data') if filepath in f]
    for f in replay_files:
        sanitize(f"src/scraper/data/{f}")
else:
    sanitize(filepath)
