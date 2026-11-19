package core.game;

import core.Types;
import core.actions.Action;
import core.actions.unitactions.Move;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/** Times operations on a flat two-capital board with the 2020 Tribes API. */
public final class PolyfishBench {
    private static final int TRACE_ROUNDS = 10;
    private static volatile Object sink;

    public static void main(String[] args) throws Exception {
        int size = Integer.parseInt(args[0]);
        int iterations = Integer.parseInt(args[1]);
        String mode = args.length > 2 ? args[2] : "ops";
        int warmup = args.length > 3 ? Integer.parseInt(args[3]) : iterations * 2;
        Path level = Files.createTempFile("polyfish-tribes-board", ".csv");
        try {
            StringBuilder map = new StringBuilder();
            for (int row = 0; row < size; row++) {
                for (int col = 0; col < size; col++) {
                    if (col > 0) map.append(',');
                    if (row == 1 && col == 1) map.append("c:0");
                    else if (row == size - 2 && col == size - 2) map.append("c:1");
                    else map.append(".:");
                }
                map.append('\n');
            }
            Files.writeString(level, map);
            GameState initial = new GameState(new Random(42), Types.GAME_MODE.CAPITALS);
            initial.init(level.toString());
            initial.initTurn(initial.getTribes()[0]);
            initial.computePlayerActions(initial.getTribes()[0]);
            Action ending = initial.getAllAvailableActions().stream()
                .filter(action -> action.getActionType() == Types.ACTION.END_TURN)
                .findFirst().orElseThrow();
            int legal = initial.getAllAvailableActions().size();
            switch (mode) {
                case "ops": {
                    Move step = findStep(initial, 2, 2);
                    run(initial, ending, step, warmup);
                    long[] times = run(initial, ending, step, iterations);
                    System.out.printf("{\"engine\":\"tribes\",\"size\":%d,\"iterations\":%d,\"legal\":%d,"
                                 + "\"cloneNs\":%d,\"viewNs\":%d,\"legalNs\":%d,\"endTurnNs\":%d,\"stepNs\":%d}%n",
                        size, iterations, legal, times[0], times[1], times[2], times[3], times[4]);
                    break;
                }
                case "trace": {
                    runTraces(initial, size, warmup);
                    long start = System.nanoTime();
                    runTraces(initial, size, iterations);
                    long elapsed = System.nanoTime() - start;
                    System.out.printf("{\"engine\":\"tribes\",\"size\":%d,\"iterations\":%d,"
                                 + "\"actionsPerTrace\":%d,\"traceNs\":%d}%n",
                        size, iterations, TRACE_ROUNDS * 4, elapsed);
                    break;
                }
                case "retain": {
                    GameState[] copies = new GameState[iterations];
                    for (int i = 0; i < copies.length; i++) copies[i] = initial.copy();
                    sink = copies;
                    System.out.printf("{\"engine\":\"tribes\",\"size\":%d,\"retainedCopies\":%d}%n",
                        size, copies.length);
                    break;
                }
                default: throw new IllegalArgumentException("Unknown benchmark mode " + mode);
            }
        } finally {
            Files.deleteIfExists(level);
        }
    }

    private static long[] run(GameState initial, Action ending, Move step, int iterations) {
        long[] times = new long[5];
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            sink = initial.copy();
            times[0] += System.nanoTime() - start;

            start = System.nanoTime();
            sink = initial.copy(0);
            times[1] += System.nanoTime() - start;

            GameState next = initial.copy();
            start = System.nanoTime();
            next.computePlayerActions(next.getActiveTribe());
            sink = next.getAllAvailableActions();
            times[2] += System.nanoTime() - start;

            start = System.nanoTime();
            next.advance(ending, false);
            times[3] += System.nanoTime() - start;
            if (next.getActiveTribeID() != 1) throw new AssertionError("EndTurn did not advance");

            GameState moving = initial.copy();
            start = System.nanoTime();
            moving.advance(step, false);
            times[4] += System.nanoTime() - start;
            if (moving.getBoard().getUnitAt(2, 2) == null) throw new AssertionError("Step did not advance");
        }
        return times;
    }

    private static Move findStep(GameState state, int x, int y) {
        for (Action action : state.getAllAvailableActions()) {
            if (action instanceof Move) {
                Move move = (Move) action;
                if (move.getDestination().x == x && move.getDestination().y == y) return move;
            }
        }
        throw new AssertionError("Scripted step missing");
    }

    private static void runTraces(GameState initial, int size, int iterations) {
        int[] capitals = {1, size - 2};
        int[] nearby = {2, size - 3};
        for (int i = 0; i < iterations; i++) {
            GameState state = initial.copy();
            int initialTick = state.getTick();
            for (int round = 0; round < TRACE_ROUNDS; round++) {
                for (int seat = 0; seat < 2; seat++) {
                    if (state.getActiveTribeID() != seat) throw new AssertionError("Unexpected seat");
                    int destination = round % 2 == 0 ? nearby[seat] : capitals[seat];
                    state.computePlayerActions(state.getActiveTribe());
                    state.advance(findStep(state, destination, destination), false);
                    if (state.getBoard().getUnitAt(destination, destination) == null)
                        throw new AssertionError("Scripted unit did not reach its destination");
                    state.computePlayerActions(state.getActiveTribe());
                    Action ending = state.getAllAvailableActions().stream()
                        .filter(action -> action.getActionType() == Types.ACTION.END_TURN)
                        .findFirst().orElseThrow();
                    state.advance(ending, false);
                }
                // The native Java game loop advances this clock outside GameState.advance.
                state.incTick();
            }
            if (state.getActiveTribeID() != 0 || state.getTick() != initialTick + TRACE_ROUNDS)
                throw new AssertionError("Incorrect final turn");
            sink = state;
        }
    }
}
