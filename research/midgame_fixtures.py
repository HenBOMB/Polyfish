"""Generate the versioned, engine-neutral midgame benchmark scenarios."""

import argparse
import json
from pathlib import Path

from run_engine_benchmark import SIDES


def scenario(side):
    middle = side // 2
    city_count = 2 if side <= 14 else 3 if side < 20 else 4
    rows = [2 + i * (side - 5) // (city_count - 1) for i in range(city_count)]
    tiles = [
        {"terrain": (
            "water" if x in (0, side - 1) and y % 4 != 2 else
            "forest" if (x * 11 + y * 7) % 13 < 3 else
            "mountain" if (x * 7 + y * 11) % 17 == 0 else "field"
        )}
        for y in range(side) for x in range(side)
    ]
    cities, units = [], []
    reserved = set()

    def index(x, y):
        return y * side + x

    def place_unit(seat, home, x, y, kind):
        location = index(x, y)
        assert location not in reserved
        tiles[location] = {"terrain": "field"}
        units.append({"seat": seat, "home": home, "tile": location, "kind": kind})
        reserved.add(location)

    for seat, x, direction in ((1, 2, 1), (2, side - 3, -1)):
        for number, y in enumerate(rows):
            city = index(x, y)
            reserved.add(city)
            tiles[city] = {"terrain": "field"}
            cities.append({"seat": seat, "tile": city, "capital": number == 0, "production": 3})
            tiles[index(x, y + 1)] = {"terrain": "field", "resource": "fruit"}
            tiles[index(x, y - 1)] = {"terrain": "forest", "resource": "game"}
            tiles[index(x + direction, y)] = {"terrain": "field", "road": True}
            place_unit(seat, city, x - direction, y, "warrior")
            place_unit(seat, city, x + direction, y - 1, "rider")

        front = middle - 1 if seat == 1 else middle
        home_index = (seat - 1) * city_count
        frontline = [
            (front, middle, "warrior"),
            (front, middle + 1, "warrior"),
            (front, middle - 1, "archer"),
        ]
        if side >= 16:
            frontline.extend([
                (front, middle - 3, "rider"),
                (front, middle + 3, "defender"),
            ])
        if side >= 20:
            frontline.extend([
                (front - direction, middle - 2, "archer"),
                (front - direction, middle + 2, "rider"),
            ])
        for number, (unit_x, unit_y, kind) in enumerate(frontline):
            place_unit(seat, cities[home_index + number % city_count]["tile"], unit_x, unit_y, kind)

    # Clear contested attack and movement lanes without clearing the fringes.
    for y in range(middle - 2, middle + 3):
        for x in range(middle - 2, middle + 3):
            tile = index(x, y)
            if tile not in reserved and "resource" not in tiles[tile]:
                tiles[tile]["terrain"] = "field"
    return {
        "version": 1, "side": side, "turn": 16, "seed": 42, "stars": 16,
        "techs": ["riding", "roads", "organization", "strategy", "hunting",
                  "archery", "climbing"],
        "tiles": tiles, "cities": cities, "units": units,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for side in SIDES:
        path = args.directory / f"midgame-{side * side}.json"
        path.write_text(json.dumps(scenario(side), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
