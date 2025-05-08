import random
import utils
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a map with specified parameters.")
    parser.add_argument('--size', type=int, default=16, help='Size of the map (default: 16)')
    parser.add_argument('--land', type=float, default=0.5, help='Initial land ratio (default: 0.5)')
    parser.add_argument('--smooth', type=int, default=3, help='Smoothing iterations (default: 3)')
    parser.add_argument('--relief', type=int, default=4, help='Relief level (default: 4)')
    parser.add_argument('--tribes', nargs='+', default=['Vengir', 'Bardur', 'Oumaji'], help='List of tribes (space-separated)')
    parser.add_argument('--seed', type=int, default=None, help='The seed to use (default: None)')
    return parser.parse_args()

def generate(map_size, initial_land, smoothing, relief, tribes, seed=None):
    if seed is not None:
        random.seed(seed)

    terrain = ['forest', 'fruit', 'game', 'ground', 'mountain']

    X2_0 = 2.0
    X1_5 = 1.5
    X1_2 = 1.2
    X1_0 = 1.0
    X0_5 = 0.5
    X0_4 = 0.4
    # X0_3 = 0.3
    X0_2 = 0.2
    X0_1 = 0.1
    X0_0 = 0.0

    BORDER_EXPANSION = 1 / 3

    # MODIFIED:
    # BARDUR GAME 2.0 -> 1.0
# Cymanti: 1.2x mountain, crop rate replaced with spore rate, cannot spawn crop.
    
    terrain_probs = {
        'water': {
            'XinXi': X0_0, 'Imperius': X0_0, 'Bardur': X0_0, 'Oumaji': X0_0, 'Kickoo': X0_4,
            'Hoodrick': X0_0, 'Luxidoor': X0_0, 'Vengir': X0_0, 'Zebasi': X0_0, 'AiMo': X0_0,
            'Quetzali': X0_0, 'Yadakk': X0_0, 'Aquarion': X1_5, 'Elyrion': X0_0, 'Cymanti': X1_0
        },
        'forest': {
            'XinXi': X1_0, 'Imperius': X1_0, 'Bardur': X1_0, 'Oumaji': X0_2, 'Kickoo': X1_0,
            'Hoodrick': X1_5, 'Luxidoor': X1_0, 'Vengir': X1_0, 'Zebasi': X0_5, 'AiMo': X1_0,
            'Quetzali': X1_0, 'Yadakk': X0_5, 'Aquarion': X0_5, 'Elyrion': X1_0, 'Cymanti': X1_0
        },
        'mountain': {
            'XinXi': X1_5, 'Imperius': X1_0, 'Bardur': X1_0, 'Oumaji': X0_5, 'Kickoo': X0_5,
            'Hoodrick': X0_5, 'Luxidoor': X1_0, 'Vengir': X1_0, 'Zebasi': X0_5, 'AiMo': X1_5,
           'Quetzali': X1_0, 'Yadakk': X0_5, 'Aquarion': X1_0, 'Elyrion': X0_5, 'Cymanti': X1_0
        },
        'metal': {
            'XinXi': X1_5, 'Imperius': X1_0, 'Bardur': X1_0, 'Oumaji': X1_0, 'Kickoo': X1_0,
            'Hoodrick': X1_0, 'Luxidoor': X1_0, 'Vengir': X2_0, 'Zebasi': X1_0, 'AiMo': X1_0,
            'Quetzali': X0_1, 'Yadakk': X1_0, 'Aquarion': X1_0, 'Elyrion': X1_0, 'Cymanti': X1_0
        },
        'fruit': {
            'XinXi': X1_0, 'Imperius': X2_0, 'Bardur': X1_5, 'Oumaji': X1_0, 'Kickoo': X1_0,
            'Hoodrick': X1_0, 'Luxidoor': X1_0, 'Vengir': X0_1, 'Zebasi': X0_5, 'AiMo': X1_0,
            'Quetzali': X2_0, 'Yadakk': X1_5, 'Aquarion': X1_0, 'Elyrion': X1_0, 'Cymanti': X1_0
        },
        'crop': {
            'XinXi': X1_0, 'Imperius': X1_0, 'Bardur': X0_1, 'Oumaji': X1_0, 'Kickoo': X1_0,
            'Hoodrick': X1_0, 'Luxidoor': X1_0, 'Vengir': X1_0, 'Zebasi': X1_0, 'AiMo': X0_1,
            'Quetzali': X0_1, 'Yadakk': X1_0, 'Aquarion': X1_0, 'Elyrion': X1_5, 'Cymanti': X0_0
        },
        'spore': {
            'XinXi': X0_0, 'Imperius': X0_0, 'Bardur': X0_0, 'Oumaji': X0_0, 'Kickoo': X0_0,
            'Hoodrick': X0_0, 'Luxidoor': X0_0, 'Vengir': X0_0, 'Zebasi': X0_0, 'AiMo': X0_0,
            'Quetzali': X0_0, 'Yadakk': X0_0, 'Aquarion': X0_0, 'Elyrion': X0_0, 'Cymanti': X1_2
        },
        'game': {
            'XinXi': X1_0, 'Imperius': X0_5, 'Bardur': X1_0, 'Oumaji': X0_2, 'Kickoo': X1_0,
            'Hoodrick': X1_0, 'Luxidoor': X1_5, 'Vengir': X0_1, 'Zebasi': X1_0, 'AiMo': X1_0,
            'Quetzali': X1_0, 'Yadakk': X1_0, 'Aquarion': X1_0, 'Elyrion': X1_0, 'Cymanti': X1_0
        },
        'fish': {
            'XinXi': X1_0, 'Imperius': X1_0, 'Bardur': X1_0, 'Oumaji': X1_0, 'Kickoo': X1_5,
            'Hoodrick': X1_0, 'Luxidoor': X1_0, 'Vengir': X0_1, 'Zebasi': X1_0, 'AiMo': X1_0,
            'Quetzali': X1_0, 'Yadakk': X1_0, 'Aquarion': X1_0, 'Elyrion': X1_0, 'Cymanti': X1_0
        },
        'starfish': {
            'XinXi': X1_0, 'Imperius': X1_0, 'Bardur': X1_0, 'Oumaji': X1_0, 'Kickoo': X1_0,
            'Hoodrick': X1_0, 'Luxidoor': X1_0, 'Vengir': X1_0, 'Zebasi': X1_0, 'AiMo': X1_0,
            'Quetzali': X1_0, 'Yadakk': X1_0, 'Aquarion': X1_0, 'Elyrion': X1_0, 'Cymanti': X1_0
        }}

    general_probs = {
        # 'mountain': 0.14, 
        # reduced for early training
        'mountain': 0.02, 
        'forest': 0.38,
        'fruit': 0.18,
        'crop': 0.18,
        'fish': 0.50,
        'game': 0.19,
        'starfish': 0.4,
        'metal': 0.5,
        # 'spore': 1.0
    }

    world_map = [{
        'type': 'ocean',
        'above': None,
        'road': False, 
        'tribe': 'XinXi',
        'otribe': 'XinXi',
    } for _ in range(map_size ** 2)]

    j = 0
    while j < map_size ** 2 * initial_land:
        cell = random.randrange(0, map_size ** 2)
        if world_map[cell]['type'] == 'ocean':
            j += 1
            world_map[cell]['type'] = 'ground'

    # disabled for early training
    land_coefficient = 1#(0.5 + relief) / 9

    for i in range(smoothing):
        for cell in range(map_size ** 2):
            water_count = 0
            tile_count = 0
            neighbours = utils.round_(cell, 1, map_size)
            for i in range(len(neighbours)):
                if world_map[neighbours[i]]['type'] == 'ocean':
                    water_count += 1
                tile_count += 1
            if water_count / tile_count <= land_coefficient:
                world_map[cell]['road'] = True
        for cell in range(map_size ** 2):
            if world_map[cell]['road']:
                world_map[cell]['road'] = False
                world_map[cell]['type'] = 'ground'
            else:
                world_map[cell]['type'] = 'ocean'

        capital_cells = []
    min_separation = 3  # no two capitals will be closer than this (in tile‐to‐tile distance)
    for tribe in tribes:
        # build a map of "valid" ground cells that are far enough from existing capitals
        capital_map = {}
        for row in range(2, map_size - 2):
            for column in range(2, map_size - 2):
                idx = row * map_size + column
                if world_map[idx]['type'] != 'ground':
                    continue

                # enforce minimum distance from every capital already placed
                too_close = False
                for cap in capital_cells:
                    if utils.distance(idx, cap, map_size) < min_separation:
                        too_close = True
                        break
                if not too_close:
                    # start its “score” high – we’ll reduce it by distance to each existing capital
                    capital_map[idx] = map_size

        # now pick the furthest‐away cell among the filtered ones
        max_dist = 0
        for cell, _ in capital_map.items():
            # compute actual min‐distance to existing capitals
            for cap in capital_cells:
                capital_map[cell] = min(capital_map[cell], utils.distance(cell, cap, map_size))
            max_dist = max(max_dist, capital_map[cell])

        # choose one of the cells whose score equals max_dist
        choices = [c for c, d in capital_map.items() if d == max_dist]
        chosen = random.choice(choices)
        capital_cells.append(chosen)
        world_map[chosen]['above'] = 'capital'
        world_map[chosen]['tribe'] = tribe
        world_map[chosen]['otribe'] = tribe


    done_tiles = []
    active_tiles = []
    for i in range(len(capital_cells)):
        done_tiles.append(capital_cells[i])
        active_tiles.append([capital_cells[i]])
        
    while len(done_tiles) != map_size ** 2:
        for i in range(len(tribes)):
            if len(active_tiles[i]) and tribes[i] != 'Polaris':
                rand_number = random.randrange(0, len(active_tiles[i]))
                rand_cell = active_tiles[i][rand_number]
                neighbours = utils.circle(rand_cell, 1, map_size)
                valid_neighbours = list(filter(lambda tile: tile not in done_tiles and
                                                    world_map[tile]['type'] != 'water', neighbours))
                if not len(valid_neighbours):
                    valid_neighbours = list(filter(lambda tile: tile not in done_tiles, neighbours))
                if len(valid_neighbours):
                    new_rand_number = random.randrange(0, len(valid_neighbours))
                    new_rand_cell = valid_neighbours[new_rand_number]
                    world_map[new_rand_cell]['tribe'] = tribes[i]
                    active_tiles[i].append(new_rand_cell)
                    done_tiles.append(new_rand_cell)
                else:
                    active_tiles[i].remove(rand_cell)

    for cell in range(map_size**2):
        if world_map[cell]['type'] == 'ground' and world_map[cell]['above'] is None:
            tribe_key = world_map[cell].get('otribe', world_map[cell]['tribe'])
            rand = random.random()
            if rand < general_probs['forest'] * terrain_probs['forest'][tribe_key]:
                world_map[cell]['type'] = 'forest'
            elif rand > 1 - general_probs['mountain'] * terrain_probs['mountain'][tribe_key]:
                world_map[cell]['type'] = 'mountain'
            rand = random.random()
            if rand < terrain_probs['water'][tribe_key]:
                world_map[cell]['type'] = 'ocean'

    village_map = []
    for cell in range(map_size**2):
        row = cell // map_size
        column = cell % map_size
        if world_map[cell]['type'] == 'ocean' or world_map[cell]['type'] == 'mountain':
            village_map.append(-1)
        elif row == 0 or row == map_size - 1 or column == 0 or column == map_size - 1:
            village_map.append(-1)
        else:
            village_map.append(0)

    land_like_terrain = ['ground', 'forest', 'mountain']
    for cell in range(map_size**2):
        if world_map[cell]['type'] == 'ocean':
            for neighbour in utils.plus_sign(cell, map_size):
                if world_map[neighbour]['type'] in land_like_terrain:
                    world_map[cell]['type'] = 'water'
                    break

    village_count = 0
    for capital in capital_cells:
        village_map[capital] = 3
        for cell in utils.circle(capital, 1, map_size):
            village_map[cell] = max(village_map[cell], 2)
        for cell in utils.circle(capital, 2, map_size):
            village_map[cell] = max(village_map[cell], 1)

    while 0 in village_map:
        new_village = random.choice(list(filter(lambda tile: True if village_map[tile] == 0 else False,
                                                list(range(len(village_map))))))
        village_map[new_village] = 3
        for cell in utils.circle(new_village, 1, map_size):
            village_map[cell] = max(village_map[cell], 2)
        for cell in utils.circle(new_village, 2, map_size):
            village_map[cell] = max(village_map[cell], 1)
        village_count += 1

    def proc(cell_, probability):
        return (village_map[cell_] == 2 and random.random() < probability) or\
            (village_map[cell_] == 1 and random.random() < probability * BORDER_EXPANSION)

    for cell in range(map_size**2):
        tribe_key = world_map[cell].get('otribe', world_map[cell]['tribe'])
        if world_map[cell]['type'] == 'ground':
            fruit = general_probs['fruit'] * terrain_probs['fruit'][tribe_key]
            crop = general_probs['crop'] * terrain_probs['crop'][tribe_key]
            spore = general_probs['crop']  * terrain_probs['spore'][tribe_key]
            if world_map[cell]['above'] != 'capital':
                if village_map[cell] == 3:
                    world_map[cell]['above'] = 'village'
                elif proc(cell, fruit * (1 - crop / 2)):
                    world_map[cell]['above'] = 'fruit'
                elif proc(cell, crop * (1 - fruit / 2)):
                    world_map[cell]['above'] = 'crop'
                elif proc(cell, spore * (1 - fruit/2)):
                    world_map[cell]['above'] = 'spore'
                elif proc(cell, crop * (1 - spore/2)):
                    world_map[cell]['above'] = 'crop'
        elif world_map[cell]['type'] == 'forest':
            if world_map[cell]['above'] != 'capital':
                if village_map[cell] == 3:
                    world_map[cell]['type'] = 'ground'
                    world_map[cell]['above'] = 'village'
                elif proc(cell, general_probs['game'] * terrain_probs['game'][tribe_key]):
                    world_map[cell]['above'] = 'game'
        elif world_map[cell]['type'] == 'water':
            if proc(cell, general_probs['fish'] * terrain_probs['fish'][tribe_key]):
                world_map[cell]['above'] = 'fish'
        elif world_map[cell]['type'] == 'ocean':
            if proc(cell, general_probs['starfish'] * terrain_probs['starfish'][tribe_key]):
                world_map[cell]['above'] = 'starfish'
        elif world_map[cell]['type'] == 'mountain':
            if proc(cell, general_probs['metal'] * terrain_probs['metal'][tribe_key]):
                world_map[cell]['above'] = 'metal'

    ruins_number = round(map_size**2/40)
    water_ruins_number = round(ruins_number/3)
    ruins_count = 0
    water_ruins_count = 0

    while ruins_count < ruins_number:
        ruin = random.choice(list(filter(lambda tile: True if village_map[tile] in (-1, 0, 1) else False,
                                                list(range(len(village_map))))))
        terrain = world_map[ruin]['type'];
        if terrain != 'water' and (water_ruins_count < water_ruins_number or terrain != 'ocean'):
            world_map[ruin]['above'] = 'ruin'  # actually there can be both ruin and resource on a single tile
            # but only ruin is displayed; as it is just a map generator it doesn't matter
            if terrain == 'ocean':
                water_ruins_count += 1
            for cell in utils.circle(ruin, 1, map_size):
                village_map[cell] = max(village_map[cell], 2)
            ruins_count += 1

    def check_resources(resource, capital):
        resources_ = 0
        for neighbour_ in utils.circle(capital, 1, map_size):
            if world_map[neighbour_]['above'] == resource:
                resources_ += 1
        return resources_

    def post_generate(resource, underneath, quantity, capital):
        resources_ = check_resources(resource, capital)
        while resources_ < quantity:
            pos_ = random.randrange(0, 8)
            territory_ = utils.circle(capital, 1, map_size)
            world_map[territory_[pos_]]['type'] = underneath
            world_map[territory_[pos_]]['above'] = resource
            for neighbour_ in utils.plus_sign(territory_[pos_], map_size):
                if world_map[neighbour_]['type'] == 'ocean':
                    world_map[neighbour_]['type'] = 'water'
            resources_ = check_resources(resource, capital)

    for capital in capital_cells:
        if world_map[capital]['tribe'] == 'Imperius':
            post_generate('fruit', 'ground', 2, capital)
        elif world_map[capital]['tribe'] == 'Bardur':
            post_generate('game', 'forest', 2, capital)
        elif world_map[capital]['tribe'] == 'Kickoo':
            resources = check_resources('fish', capital)
            while resources < 2:
                pos = random.randrange(0, 4)
                territory = utils.plus_sign(capital, map_size)
                world_map[territory[pos]]['type'] = 'water'
                world_map[territory[pos]]['above'] = 'fish'
                for neighbour in utils.plus_sign(territory[pos], map_size):
                    if world_map[neighbour]['type'] == 'water':
                        world_map[neighbour]['type'] = 'ocean'
                        for double_neighbour in utils.plus_sign(neighbour, map_size):
                            if world_map[double_neighbour]['type'] != 'water' and world_map[double_neighbour]['type'] != 'ocean':
                                world_map[neighbour]['type'] = 'water'
                                break
                resources = check_resources('fish', capital)
            break
        elif world_map[capital]['tribe'] == 'Zebasi':
            post_generate('crop', 'ground', 1, capital)
        elif world_map[capital]['tribe'] == 'Elyrion':
            post_generate('game', 'forest', 2, capital)
        elif world_map[capital]['tribe'] == 'Polaris':
            for neighbour in utils.circle(capital, 1, map_size):
                world_map[neighbour]['tribe'] = 'Polaris'

    return world_map

if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(
        generate(
            args.size,
            args.land,
            args.smooth,
            args.relief,
            args.tribes,
            args.seed
        )
    ))
