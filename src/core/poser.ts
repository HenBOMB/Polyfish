// import { mkdirSync, existsSync } from "fs";
// import Database from "better-sqlite3";
// import { MoveGenerator } from "./moves";
// import Game from "../game";
// import Move from "./move";
// import { GameState } from "./states";

// export default class PoseManager {
//     private db: Database.Database;
//     /** raw JSON → { state, moves } */
//     private cache = new Map<
//         GameState,
//         Move[]
//     >();
//     /** new entries to write at save() time */
//     private pending: Array<{
//         state: GameState;
//         moves: Move[];
//     }> = [];
//     private insertStmt: Database.Statement;
//     private inTransaction = false;

//     constructor(
//         dbPath: string = "data/poses.sqlite",
//         usePragmas: boolean = true
//     ) {
//         // ensure directory exists
//         const dir = dbPath.replace(/\/[^/]+$/, "");
//         if (dir && !existsSync(dir)) mkdirSync(dir, { recursive: true });

//         this.db = new Database(dbPath);
//         if (usePragmas) {
//             this.db.pragma("journal_mode = MEMORY");
//             this.db.pragma("synchronous = OFF");
//             this.db.pragma("cache_size = 10000");
//         }

//         // state TEXT is the JSON of the GameState; PRIMARY KEY avoids dupes
//         this.db.exec(`
//         CREATE TABLE IF NOT EXISTS poses (
//             state TEXT PRIMARY KEY,
//             moves TEXT NOT NULL
//         )
//         `);

//         this.insertStmt = this.db.prepare(
//         `INSERT OR IGNORE INTO poses (state, moves) VALUES (?, ?)`
//         );

//         // load everything into `cache`
//         this.load();
//     }

//     /** Pull every saved state+moves into memory */
//     private load() {
//         this.cache.clear();
//         for (const row of this.db
//             .prepare(`SELECT state, moves FROM poses`)
//             .iterate() as IterableIterator<{ state: string; moves: string }>
//         ) {
//             const parsedState = JSON.parse(row.state) as GameState;
//             const actions = row.moves.split("#").map(Move.deserialize);
//             const moves = MoveGenerator.fromActions(actions);
//             this.cache.set(parsedState, moves);
//         }
//         console.log(`Loaded ${this.cache.size} poses from database`);
//     }

//     /**
//      * Get the list of legal moves for this game-state.
//      * Uses in-memory cache → recompute & queue for save() if missing.
//      */
//     get(game: Game): Move[] {
//         if (this.cache.has(game.state)) {
//             return this.cache.get(game.state)!;
//         }

//         const moves = MoveGenerator.legal(game.state);
        
//         this.cache.set(game.state, moves);

//         this.pending.push({ state: game.state, moves });

//         return moves;
//     }

//     /**
//      * Persist _all_ newly seen states & their moves to SQLite.
//      * Wraps them in a single transaction for speed.
//      */
//     save() {
//         if (this.pending.length === 0) return;
//         this.db.exec("BEGIN");
//         try {
//             for (const { state, moves } of this.pending) {
//                 this.insertStmt.run(JSON.stringify(state), MoveGenerator.serialize(moves));
//             }
//             this.db.exec("COMMIT");
//             this.pending = [];
//         } catch (err) {
//             this.db.exec("ROLLBACK");
//             throw err;
//         }
//     }

//     /** Make sure to flush any open transaction */
//     close() {
//         this.save();
//         this.db.close();
//     }
// }

import { mkdirSync, existsSync } from "fs";
import Database from "better-sqlite3";
import { MoveGenerator } from "./moves";
import Game from "../game";
import Move from "./move";
import { createHash } from "crypto";

export default class PoseManager {
    private db: Database.Database;
    private poses: Set<string>;
    private inMemory: Map<string, Move[]>;
    private insertStmt: Database.Statement;
    private selectStmt: Database.Statement;

    /**
     * @param dbPath
     * @param usePragmas
     */
    constructor(
        dbPath: string = "data/poses.sqlite",
        usePragmas: boolean = true
    ) {
        const dir = dbPath.replace(/\/[^/]+$/, "");
        if (dir && !existsSync(dir)) mkdirSync(dir, { recursive: true });

        this.db = new Database(dbPath);

        if (usePragmas) {
            this.db.pragma("journal_mode = MEMORY");
            this.db.pragma("synchronous = OFF");
            this.db.pragma("cache_size = 10000");
        }

        // create table if needed
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS poses (
                hash   TEXT PRIMARY KEY,
                moves  TEXT NOT NULL
            )
        `);

        // prepare and cache statements
        this.insertStmt = this.db.prepare(`INSERT OR IGNORE INTO poses (hash, moves) VALUES (?, ?)`);
        this.selectStmt = this.db.prepare(`SELECT moves FROM poses WHERE hash = ?`);

        // in‐mem structures
        this.poses = new Set<string>();
        this.inMemory = new Map<string, Move[]>();

        this.load();
  }

    /** Load all known hashes into memory (optional warmup) */
    load() {
        this.poses.clear();
        for (const row of this.db.prepare(`SELECT hash FROM poses`).iterate()) {
            this.poses.add((row as any).hash);
        }
        console.log(`Loaded ${this.poses.size} poses from database`);
        this.db.exec("BEGIN");
    }

    /**
     * Get the list of legal moves for this game-state.
     * Uses in-memory cache → SQLite cache → recompute & store.
     */
    get(game: Game): Move[] {
        // game.xxh.h64(game.state, 0);
        // const key = this.hashGameState(game);
        // // 1) In-memory hot cache
        // const hot = this.inMemory.get(key);
        // if (hot) return hot;

        // // 2) SQLite cache
        // if (this.poses.has(key)) {
        //     const row = this.selectStmt.get(key) as { moves: string } | undefined;
        //     if (row) {
        //         const actions = row.moves.split("#").map(Move.deserialize);
        //         const moves = MoveGenerator.fromActions(actions);
        //         this.inMemory.set(key, moves);
        //         return moves;
        //     }
        //     // fall through if inconsistency
        // }

        // // 3) Miss → compute, store, cache
        const moves = MoveGenerator.legal(game.state);
        // const serialized = MoveGenerator.serialize(moves); // e.g. "m1#m2#..."
        // this.insertStmt.run(key, serialized);
        // this.poses.add(key);
        // this.inMemory.set(key, moves);
        return moves;
    }

    /** Close DB (commit any open tx and free resources) */
    close() {
        this.db.exec("COMMIT");
        this.db.close();
    }

    /** SHA-256 of the JSON state */
    private hashGameState(game: Game): string {
        return createHash("sha256")
        .update(JSON.stringify(game.state))
        .digest("hex");
    }
}
