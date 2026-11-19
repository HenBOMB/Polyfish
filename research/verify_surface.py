"""Check full matches, a common browser/headless turn, and replay playback."""

import argparse
from copy import deepcopy
from functools import partial
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen


MOVE_TYPES = {
    "step": 1, "attack": 2, "ability": 3, "summon": 4, "upgrade": 4,
    "harvest": 5, "build": 6, "research": 7, "capture": 8, "reward": 9, "endTurn": 10,
}


def browser_key(command):
    key = {"moveType": MOVE_TYPES[command["type"]]}
    target_field = "src" if command["type"] == "summon" else "target"
    for source, target in [("source", "src"), ("target", target_field), ("technology", "type"),
                           ("structure", "type"), ("unit", "type"), ("ability", "type"), ("reward", "type")]:
        if source in command:
            key[target] = command[source]
    return key


def request_json(base_url, path, data=None):
    body = None if data is None else json.dumps(data).encode()
    try:
        with urlopen(Request(base_url + path, body, {"Content-Type": "application/json"})) as response:
            return json.load(response)
    except HTTPError as error:
        error.add_note(error.read().decode())
        raise


def verify_save_rejects_invalid_turn_order(base_url):
    request = partial(request_json, base_url)
    before = request("/reset", {"mapSize": 121, "seed": 42, "opponentPreset": "normal"})
    saved = request("/research/save")
    invalid_saves = []
    for key, value in [("_maxTribeCount", 1), ("currentPlayerTurnId", 3)]:
        invalid = deepcopy(saved)
        invalid["settings"][key] = value
        invalid_saves.append(invalid)
    invalid = deepcopy(saved)
    invalid["tribes"]["3"] = invalid["tribes"].pop("2")
    invalid["tribes"]["3"]["id"] = 3
    invalid_saves.append(invalid)
    invalid = deepcopy(saved)
    invalid["settings"]["research"]["botSeat"] = 3
    invalid_saves.append(invalid)

    for invalid in invalid_saves:
        try:
            request("/research/load", invalid)
        except HTTPError as error:
            assert error.code == 400, error
        else:
            raise AssertionError("Save with invalid turn settings was accepted")
        assert request("/research/current") == before
    after = request("/research/move", {"revision": before["revision"], "action": {"moveType": 10}})
    assert after["active"] == 2 and after["revision"] == before["revision"] + 1


def verify(base_url, binary, size, preset, output, full_api):
    request = partial(request_json, base_url)

    def play_human(game, command):
        action = browser_key(command)
        excavates_ruin = command["type"] == "capture" and any(
            tile["index"] == action["src"] and tile.get("structure") == "Ruin" for tile in game["tiles"]
        )
        played = request("/research/move", {"revision": game["revision"], "action": action})
        if excavates_ruin:
            assert any(message.startswith("Ruin reward:") for message in played.get("messages", []))
            assert "messages" not in request("/research/current")
        return played

    result = json.loads(subprocess.check_output([
        str(binary), "--map-size", str(size), "--preset1", "normal", "--preset2", preset,
        "--seed", "42", "--replay", str(output),
    ], text=True))
    replay = json.loads(output.read_text())
    game = request("/reset", {
        "mapSize": size, "seed": 42, "opponentPreset": preset,
        "tribe1": "imperius", "tribe2": "bardur",
    })
    assert game["seed"] == "42" and len(game["tiles"]) == size
    assert any(tile == {"index": tile["index"], "visible": False} for tile in game["tiles"])
    try:
        request("/reset", {"mapSize": 400})
    except HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError("Invalid map size accepted")
    assert request("/research/current") == game

    # Replay the human seat's first turn, then ask the browser API to play its opponent.
    for command in replay["turns"][0]["commands"]:
        game = play_human(game, command)
    start = game["revision"]
    saved = request("/research/save")
    for pass_index in range(2):
        actions = 0
        while game["botTurn"]:
            game = request("/research/bot", {})
            actions += 1
            assert actions <= 100
        expected = [browser_key(command) for command in replay["turns"][1]["commands"]]
        assert request("/research/save")["history"][start:] == expected
        if pass_index == 0:
            game = request("/research/load", saved)

    if full_api:
        for replay_turn in replay["turns"][2:]:
            assert game["turn"] == replay_turn["turnNumber"]
            assert game["active"] == replay_turn["playerId"]
            if game["botTurn"]:
                start = game["revision"]
                for _ in replay_turn["commands"]:
                    assert game["botTurn"], "Browser bot finished before the replay turn"
                    game = request("/research/bot", {})
                expected = [browser_key(command) for command in replay_turn["commands"]]
                assert request("/research/save")["history"][start:] == expected
            else:
                for command in replay_turn["commands"]:
                    game = play_human(game, command)
        assert game["result"] == result["result"]
    playback = request("/research/replay", replay)
    final = request("/research/replay/seek", {"position": playback["replay"]["total"]})
    assert final["result"] == result["result"]
    print(json.dumps(dict(result, browserBotActions=actions, preset=preset)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary", type=Path)
    p.add_argument("--server", default="http://127.0.0.1:3000")
    p.add_argument("--full-api", action="store_true", help="Replay all decisions through the live browser API")
    args = p.parse_args()
    verify_save_rejects_invalid_turn_order(args.server)
    with tempfile.TemporaryDirectory(prefix="polyfish-smoke-") as tmp:
        for size, preset in itertools.product((121, 196, 256), ("easy", "normal", "expert")):
            verify(args.server, args.binary.resolve(), size, preset, Path(tmp) / "match.json", args.full_api)


if __name__ == "__main__":
    main()
