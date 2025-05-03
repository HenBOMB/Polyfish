import { appendFileSync, writeFileSync } from "fs";
import Move, { MoveType } from "../core/move";
import { TribeState } from "../core/states";
import { TribeType } from "../core/types";

export const DEFAULT_LOG_FILE = 'training.poly.log';
export const DEFAULT_TIME_FORMAT = 'HH:mm:ss';

function getTimestamp(): string {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    return `${hours}:${minutes}:${seconds}`;
}

export class Logger {
    static clear(): void {
        writeFileSync(DEFAULT_LOG_FILE, '');
    }
    
    static log(message: string): null {
        appendFileSync(DEFAULT_LOG_FILE, `${getTimestamp()} - ${message}\n`);
        return null;
    }

    static logPlay(tribe: TribeState, score: number, ...moves: Move[]): null {
        const name = TribeType[tribe.tribeType];
        appendFileSync(DEFAULT_LOG_FILE, `${tribe.owner} - ${name} - ${moves.map(x => x.stringify()).join(', ')} (${score})\n`);
        return null;
    }

    static warn(message: string): null {
        appendFileSync(DEFAULT_LOG_FILE, `${getTimestamp()} - ${message}\n`);
        return null;
    }

    static error(message: string | { error: string }): null {
        appendFileSync(DEFAULT_LOG_FILE, `${getTimestamp()} - ERROR - ${typeof message === 'string' ? message : message.error}\n`);
        return null;
    }

    static illegal(moveType: MoveType, reason: string): null {
        appendFileSync(DEFAULT_LOG_FILE, `${getTimestamp()} - ILLEGAL - ${moveType != MoveType.None? MoveType[moveType] + ' ' : ''}${reason}\n`);
        return null;
    }
}
