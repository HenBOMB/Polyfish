import argparse, re, os

MAGIC = r"https://share\.polytopia\.io/[gl]/\w{8}-\w{4}-\w{4}-\w{4}-\w{12}"

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, help='input file path')
parser.add_argument('-o', type=str, help='output file path')
parser.add_argument('-c', type=bool, help='combine?', default=False)

args = parser.parse_args()
dir_input: str = args.i
dir_output: str = args.o
combine: bool = args.c

if dir_input == dir_output:
    print("Input and output file cannot be the same")
    exit(1)

data = open(dir_input, "r").read()

# extract valid replay urls from messy data

replay_urls: list[str] = list(set(re.findall(MAGIC, data)))

if os.path.exists(dir_output) or combine:
    print(f"Found {len(replay_urls)} replays")
    if combine:
        print(f"Combining urls from both files..")
    else: 
        print(f"Warning: output file {dir_output} already exists, merging both files")
    
    other_urls = open(dir_output, "r").read().split("\n")

    replay_urls.extend(other_urls)

    replay_urls = list(set(replay_urls))

open(dir_output, "w").write("\n".join(replay_urls))

print(f"Saved {len(replay_urls)} replays")
