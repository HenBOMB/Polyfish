import re

with open("polyfish-rs/src/bin/self_play.rs", "r") as f:
    content = f.read()

# 1. First conflict (~199): dump_failed_game + curriculum
p1 = r"<<<<<<< HEAD\n(.*?)match serde_json::to_vec\(&wrapped\) \{(.*?)\s+Err\(e\) => eprintln!\(\"\[dump-failed\] failed to serialize decisions: \{e\}\"\),\n    \}\n=======\n(.*?)\n>>>>>>> main\n"
def repl1(m):
    return m.group(1) + "match serde_json::to_vec(&wrapped) {" + m.group(2) + "\n        Err(e) => eprintln!(\"[dump-failed] failed to serialize decisions: {e}\"),\n    }\n}\n" + m.group(3) + "\n"
content = re.sub(p1, repl1, content, flags=re.DOTALL)

# 2. Second conflict (~463): winner_score, recap, etc.
p2 = r"<<<<<<< HEAD\n    recap: Replay,\n=======\n(.*?)    recap: ModReplay,\n>>>>>>> main\n"
def repl2(m):
    return m.group(1) + "    recap: Replay,\n"
content = re.sub(p2, repl2, content, flags=re.DOTALL)

# 3. Third conflict (~1066 & GameResult creation): 
# HEAD added the recap definition, main moved it into GameResult.
p3 = r"<<<<<<< HEAD\n(.*?)\n=======\n>>>>>>> main\n(.*?)        anchor_seat: match \((.*?)\},\n        recap: ModReplay \{\n            game_state: initial_state,\n            turns: group_recap\(flat_recap\),\n        \},(.*?)    \}\)\n\}"
def repl3(m):
    return m.group(1) + "\n" + m.group(2) + "        anchor_seat: match (" + m.group(3) + "},\n        recap,\n" + m.group(4) + "    })\n}"
content = re.sub(p3, repl3, content, flags=re.DOTALL)

# 4. Fourth conflict (~1184): group_recap signature and anchor history
p4 = r"<<<<<<< HEAD\nfn group_recap\(flat: Vec<\(i32, i32, ReplayCommand\)>\) -> Vec<ReplayTurn> \{\n=======\n(.*?)fn group_recap\(flat: Vec<\(i32, i32, serde_json::Value\)>\) -> Vec<ReplayTurn> \{\n>>>>>>> main"
def repl4(m):
    return m.group(1) + "fn group_recap(flat: Vec<(i32, i32, ReplayCommand)>) -> Vec<ReplayTurn> {"
content = re.sub(p4, repl4, content, flags=re.DOTALL)

with open("polyfish-rs/src/bin/self_play.rs", "w") as f:
    f.write(content)
