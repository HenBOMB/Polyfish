package core.game;

import core.Types;
import core.actions.Action;
import core.actions.cityactions.ResourceGathering;
import core.actions.unitactions.Attack;
import core.actions.unitactions.Move;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import org.json.JSONArray;
import org.json.JSONObject;
import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;
import utils.ElapsedCpuTimer;
import utils.Vector2d;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/** The Java adapter for the checked-in midgame fixtures. */
public final class PolyfishMidgameBench {
    private static final int STEP = 0, ATTACK = 1, HARVEST = 2, END = 3;
    private static final int NATIVE_ITERATIONS = 512;
    private static volatile Object sink;

    record Descriptor(int kind, int source, int target) {
        static Descriptor from(JSONArray row) {
            return new Descriptor(row.getInt(0), row.getInt(1), row.getInt(2));
        }

        JSONArray json() {
            return new JSONArray().put(kind).put(source).put(target);
        }
    }

    record Branch(int parent, Descriptor action) {}

    private static JSONObject object(JSONArray array, int index) {
        return array.getJSONObject(index);
    }

    private static GameState root(JSONObject scenario, int seat, boolean fog) {
        Random rng = new Random(scenario.getLong("seed"));
        Tribe[] tribes = {new Tribe(Types.TRIBE.XIN_XI), new Tribe(Types.TRIBE.IMPERIUS)};
        Board board = new Board();
        int side = scenario.getInt("side");
        board.init(side, tribes);

        JSONArray tiles = scenario.getJSONArray("tiles");
        for (int i = 0; i < tiles.length(); i++) {
            JSONObject tile = object(tiles, i);
            int x = i % side, y = i / side;
            board.setTerrainAt(x, y, switch (tile.getString("terrain")) {
                case "field" -> Types.TERRAIN.PLAIN;
                case "forest" -> Types.TERRAIN.FOREST;
                case "mountain" -> Types.TERRAIN.MOUNTAIN;
                case "water" -> Types.TERRAIN.SHALLOW_WATER;
                default -> throw new IllegalArgumentException("Unknown fixture terrain");
            });
            if (tile.has("resource")) {
                board.setResourceAt(x, y, switch (tile.getString("resource")) {
                    case "fruit" -> Types.RESOURCE.FRUIT;
                    case "game" -> Types.RESOURCE.ANIMAL;
                    default -> throw new IllegalArgumentException("Unknown fixture resource");
                });
            }
        }
        JSONArray cities = scenario.getJSONArray("cities");
        HashMap<Integer, City> byTile = new HashMap<>();
        for (int i = 0; i < cities.length(); i++) {
            JSONObject spec = object(cities, i);
            int idx = spec.getInt("tile");
            City city = new City(idx % side, idx / side, spec.getInt("seat") - 1);
            city.setCapital(spec.getBoolean("capital"));
            city.levelUp();
            city.levelUp();
            city.setPopulation(0);
            board.setTerrainAt(idx % side, idx / side, Types.TERRAIN.CITY);
            board.addCityToTribe(city, rng);
            board.assignCityTiles(city, city.getBound());
            byTile.put(idx, city);
        }
        JSONArray units = scenario.getJSONArray("units");
        for (int i = 0; i < units.length(); i++) {
            JSONObject spec = object(units, i);
            int idx = spec.getInt("tile");
            City city = byTile.get(spec.getInt("home"));
            TypeCheck.require(city.getTribeId() == spec.getInt("seat") - 1, "Unit home belongs to other tribe");
            Types.UNIT kind = Types.UNIT.valueOf(spec.getString("kind").toUpperCase(java.util.Locale.ROOT));
            Unit unit = Types.UNIT.createUnit(new Vector2d(idx % side, idx / side), 0, false,
                    city.getActorId(), city.getTribeId(), kind);
            board.addUnit(city, unit);
            TypeCheck.require(city.getUnitsID().contains(unit.getActorId()), "Unit exceeded home city capacity");
            unit.setStatus(Types.TURN_STATUS.FRESH);
        }
        for (int i = 0; i < tiles.length(); i++) {
            if (object(tiles, i).optBoolean("road", false)) board.addRoad(i % side, i / side);
        }
        for (int i = 0; i < cities.length(); i++) {
            JSONObject spec = object(cities, i);
            City city = byTile.get(spec.getInt("tile"));
            city.setProduction(0);
            city.setProduction(spec.getInt("production") - city.getProduction());
        }
        for (int id = 0; id < tribes.length; id++) {
            Tribe tribe = tribes[id];
            JSONArray techs = scenario.getJSONArray("techs");
            for (int i = 0; i < techs.length(); i++) {
                String name = techs.getString(i).toUpperCase(java.util.Locale.ROOT);
                Types.TECHNOLOGY tech = Types.TECHNOLOGY.valueOf(name.equals("STRATEGY") ? "SHIELDS" : name);
                if (!tribe.getTechTree().isResearched(tech)) {
                    TypeCheck.require(tribe.getTechTree().doResearch(tech), "Tech prerequisite missing " + tech);
                }
            }
            tribe.setStars(scenario.getInt("stars"));
            if (!tribe.getTribesMet().contains(1 - id)) tribe.getTribesMet().add(1 - id);
            boolean[][] visibility = tribe.getObsGrid();
            for (int x = 0; x < side; x++) {
                boolean visible = !fog || (id == 0 ? x <= side / 2 + 1 : x >= side / 2 - 2);
                Arrays.fill(visibility[x], visible);
            }
        }
        // The save constructor advances the incoming seat and increments the round on wrap.
        board.setActiveTribeID(seat == 1 ? 1 : 0);
        return new GameState(rng, Types.GAME_MODE.CAPITALS, tribes, board,
                scenario.getInt("turn") - (seat == 1 ? 1 : 0));
    }

    private static int index(Vector2d pos, int side) {
        return pos.y * side + pos.x;
    }

    private static Descriptor descriptor(GameState game, Action action, int side) {
        return switch (action.getActionType()) {
            case MOVE -> {
                Move move = (Move) action;
                yield new Descriptor(STEP, index(game.getActor(move.getUnitId()).getPosition(), side),
                        index(move.getDestination(), side));
            }
            case ATTACK -> {
                Attack attack = (Attack) action;
                yield new Descriptor(ATTACK, index(game.getActor(attack.getUnitId()).getPosition(), side),
                        index(game.getActor(attack.getTargetId()).getPosition(), side));
            }
            case RESOURCE_GATHERING -> {
                ResourceGathering gather = (ResourceGathering) action;
                yield new Descriptor(HARVEST, -1, index(gather.getTargetPos(), side));
            }
            case END_TURN -> new Descriptor(END, -1, -1);
            default -> null;
        };
    }

    private static JSONObject projection(GameState game, int side) {
        Board board = game.getBoard();
        JSONArray cities = new JSONArray(), units = new JSONArray(), land = new JSONArray();
        for (int idx = 0; idx < side * side; idx++) {
            int x = idx % side, y = idx / side;
            Types.TERRAIN terrain = board.getTerrainAt(x, y);
            City city = board.getCityInBorders(x, y);
            Unit unit = board.getUnitAt(x, y);
            if (terrain == Types.TERRAIN.CITY) {
                cities.put(new JSONArray().put(idx).put(city.getTribeId() + 1)
                        .put(city.getPopulation()).put(city.getProduction()));
            }
            if (unit != null) {
                int kind = switch (unit.getType()) {
                    case WARRIOR -> 0;
                    case RIDER -> 1;
                    case ARCHER -> 2;
                    case DEFENDER -> 3;
                    default -> -1;
                };
                units.put(new JSONArray().put(idx).put(unit.getTribeId() + 1).put(kind).put(unit.getCurrentHP()));
            }
            int kind = switch (terrain) {
                case PLAIN, CITY -> 0;
                case FOREST -> 1;
                case MOUNTAIN -> 2;
                case SHALLOW_WATER -> 3;
                default -> -1;
            };
            Types.RESOURCE resource = board.getResourceAt(x, y);
            int item = resource == Types.RESOURCE.FRUIT ? 1 : resource == Types.RESOURCE.ANIMAL ? 2 : 0;
            land.put(new JSONArray().put(kind).put(city == null ? 0 : city.getTribeId() + 1)
                    .put(item).put(board.isRoad(x, y) ? 1 : 0));
        }
        return new JSONObject().put("seat", game.getActiveTribeID() + 1).put("turn", game.getTick())
                .put("stars", new JSONArray().put(game.getTribe(0).getStars()).put(game.getTribe(1).getStars()))
                .put("cities", cities).put("units", units).put("land", land);
    }

    private static GameState expand(GameState parent, Descriptor wanted, int side, long[] times) {
        long start = times == null ? 0 : System.nanoTime();
        GameState child = parent.copy();
        if (times != null) times[0] += System.nanoTime() - start;
        start = times == null ? 0 : System.nanoTime();
        child.computePlayerActions(child.getActiveTribe());
        ArrayList<Action> legal = child.getAllAvailableActions();
        if (times != null) times[1] += System.nanoTime() - start;
        start = times == null ? 0 : System.nanoTime();
        Action found = null;
        for (Action action : legal) {
            if (wanted.equals(descriptor(child, action, side))) {
                found = action;
                break;
            }
        }
        if (times != null) times[2] += System.nanoTime() - start;
        if (found == null) throw new IllegalStateException("Missing " + wanted);
        start = times == null ? 0 : System.nanoTime();
        int seat = child.getActiveTribeID();
        child.advance(found, false);
        if (wanted.kind == END && child.getActiveTribeID() < seat) child.incTick();
        if (times != null) times[3] += System.nanoTime() - start;
        return child;
    }

    private static void oracle(GameState root, int side) throws Exception {
        List<GameState> states = new ArrayList<>();
        states.add(root);
        BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = input.readLine()) != null) {
            JSONObject command = new JSONObject(line);
            GameState parent = states.get(command.getInt("node"));
            JSONObject response;
            switch (command.getString("op")) {
                case "inspect" -> {
                    // Match timed expansion, where copy invalidates action computation.
                    GameState game = parent.copy();
                    game.computePlayerActions(game.getActiveTribe());
                    List<Action> legal = game.getAllAvailableActions();
                    JSONArray desc = new JSONArray();
                    for (Action action : legal) {
                        Descriptor result = descriptor(game, action, side);
                        if (result != null) desc.put(result.json());
                    }
                    response = new JSONObject().put("projection", projection(parent, side))
                            .put("legal", legal.size()).put("actions", desc);
                }
                case "expand" -> {
                    GameState child = expand(parent, Descriptor.from(command.getJSONArray("action")), side, null);
                    int slot = command.getInt("slot");
                    if (slot == states.size()) states.add(child);
                    else states.set(slot, child);
                    response = projection(child, side);
                }
                default -> throw new IllegalArgumentException("Unknown oracle command");
            }
            System.out.println(response);
        }
    }

    private static void search(GameState root, Branch[] schedule, int limit, int side, long[] times) {
        GameState[] held = new GameState[limit + 1];
        held[0] = root.copy();
        for (int i = 0; i < limit; i++) {
            Branch branch = schedule[i];
            held[i + 1] = expand(held[branch.parent], branch.action, side, times);
        }
        sink = held;
        sink = null;
    }

    private static void nativeSearch(GameState[] roots, int count, long seed) {
        MCTSParams params = new MCTSParams();
        params.stop_type = params.STOP_ITERATIONS;
        params.num_iterations = NATIVE_ITERATIONS;
        params.ROLLOUT_LENGTH = 20;
        ArrayList<Integer> seats = new ArrayList<>(List.of(0, 1));
        for (int i = 0; i < count; i++) {
            int seat = i % roots.length;
            GameState view = roots[seat].copy(seat);
            view.computePlayerActions(view.getActiveTribe());
            TypeCheck.require(view.getAllAvailableActions().size() > 1, "No MCTS decision");
            MCTSPlayer agent = new MCTSPlayer(seed + i, params);
            agent.setPlayerIDs(seat, seats);
            Action action = agent.act(view, new ElapsedCpuTimer());
            TypeCheck.require(action != null, "MCTS did not select an action");
            sink = action;
        }
    }

    public static void main(String[] args) throws Exception {
        JSONObject scenario = new JSONObject(Files.readString(Path.of(args[0])));
        TypeCheck.require(scenario.getInt("version") == 1, "Unsupported fixture version");
        int side = scenario.getInt("side");
        String mode = args[1];
        if (mode.equals("oracle")) {
            oracle(root(scenario, 1, false), side);
            return;
        }
        int count = Integer.parseInt(args[2]), warmup = Integer.parseInt(args[3]);
        JSONObject result = new JSONObject();
        switch (mode) {
            case "matched", "components", "retain" -> {
                JSONArray nodes = new JSONArray(Files.readString(Path.of(args[4])));
                Branch[] schedule = new Branch[nodes.length()];
                for (int i = 0; i < nodes.length(); i++) {
                    JSONObject node = object(nodes, i);
                    schedule[i] = new Branch(node.getInt("parent"), Descriptor.from(node.getJSONArray("action")));
                }
                GameState start = root(scenario, 1, false);
                if (mode.equals("retain")) {
                    search(start, schedule, count, side, null);
                    result.put("retained", count);
                } else {
                    for (int i = 0; i < warmup; i++) search(start, schedule, 512, side, null);
                    long[] times = mode.equals("components") ? new long[4] : null;
                    long begin = System.nanoTime();
                    for (int i = 0; i < count; i++) search(start, schedule, 512, side, times);
                    result.put("nanos", System.nanoTime() - begin).put("expansions", count * 512);
                    if (times != null) {
                        result.put("copyNs", times[0]).put("legalNs", times[1])
                                .put("lookupNs", times[2]).put("applyNs", times[3]);
                    }
                }
            }
            case "native" -> {
                GameState[] roots = {root(scenario, 1, true), root(scenario, 2, true)};
                long seed = scenario.getLong("seed");
                nativeSearch(roots, warmup, seed);
                long begin = System.nanoTime();
                nativeSearch(roots, count, seed + warmup);
                result.put("nanos", System.nanoTime() - begin).put("decisions", count)
                        .put("iterationsPerDecision", NATIVE_ITERATIONS);
            }
            default -> throw new IllegalArgumentException("Unknown benchmark mode " + mode);
        }
        System.out.println(result.put("engine", "tribes").put("size", side));
    }

    private static final class TypeCheck {
        static void require(boolean valid, String message) {
            if (!valid) throw new IllegalStateException(message);
        }
    }
}
