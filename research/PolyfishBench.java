package core.game;

import core.Types;
import core.actions.Action;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/** Times operations on a flat two-capital board with the 2020 Tribes API. */
public final class PolyfishBench {
    private static volatile Object sink;

    public static void main(String[] args) throws Exception {
        int size = Integer.parseInt(args[0]);
        int iterations = Integer.parseInt(args[1]);
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
            run(initial, ending, iterations * 2);
            long[] timings = run(initial, ending, iterations);
            System.out.printf("{\"engine\":\"tribes\",\"size\":%d,\"iterations\":%d,\"legal\":%d,"
                         + "\"cloneNs\":%d,\"viewNs\":%d,\"legalNs\":%d,\"endTurnNs\":%d}%n",
                size, iterations, legal, timings[0], timings[1], timings[2], timings[3]);
        } finally {
            Files.deleteIfExists(level);
        }
    }

    private static long[] run(GameState initial, Action ending, int iterations) {
        long[] times = new long[4];
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
        }
        return times;
    }
}
