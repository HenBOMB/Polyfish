import { mkdirSync, existsSync } from "fs";
import Database from "better-sqlite3";
import { MoveGenerator } from "./moves";
import Game from "../game";
import Move from "./move";
import { getPovTribe } from "./functions";

export default class PoseManager {
    inMemory: Map<string, Move[]>;
    pending: Map<string, Move[]>;
    private db: Database.Database;
    private poses: Set<string>;
    private insertStmt: Database.Statement;
    private selectStmt: Database.Statement;

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
            this.db.pragma("cache_size = 10000000");
        }

        this.db.exec(`
            CREATE TABLE IF NOT EXISTS poses (
                hash   TEXT PRIMARY KEY,
                moves  TEXT NOT NULL
            )
        `);

        this.insertStmt = this.db.prepare(`INSERT OR IGNORE INTO poses (hash, moves) VALUES (?, ?)`);
        this.selectStmt = this.db.prepare(`SELECT moves FROM poses WHERE hash = ?`);

        this.poses = new Set<string>();
        this.inMemory = new Map<string, Move[]>();
        this.pending = new Map<string, Move[]>();
    }

    load() {
        this.inMemory.clear();
        for (const row of this.db.prepare(`SELECT hash, moves FROM poses`).iterate()) {
            const key = (row as any).hash;
            this.poses.add(key);
            // const actions = (row as any).moves.split("#").map(Move.deserialize);
            // const moves = MoveGenerator.fromActions(actions);
            // this.inMemory.set(key, moves);
        }
        console.log(`Loaded ${this.poses.size} poses from database`);
        this.db.exec("BEGIN");
    }

    get(game: Game): Move[] {
        const key = getPovTribe(game.state).hash.toString();
        const hot = this.inMemory.get(key);

        if(hot) {
            return hot;
        }

        if(this.poses.has(key)) {
            const row: any = this.selectStmt.get(key);
            if(row) {
                const actions = row.moves.split("#").map(Move.deserialize);
                const moves = MoveGenerator.fromActions(actions);
                this.inMemory.set(key, moves);
                return moves;
            }
            // fall through if inconsistency
        }

        const moves = MoveGenerator.legal(game.state);

        if(game.state.settings._pendingRewards.length) {
            return moves;
        }
        
        this.inMemory.set(key, moves);
        this.pending.set(key, moves);

        return moves;
    }

    close() {
        for (const [key, moves] of this.pending.entries()) {
            this.insertStmt.run(key, MoveGenerator.serialize(moves));
        }
        this.db.exec("COMMIT");
        this.db.close();
    }
}
