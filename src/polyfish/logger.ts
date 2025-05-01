import { appendFileSync, existsSync } from "fs";
import { MoveType } from "../core/move";

export const DEFAULT_LOG_FILE = 'training.log';
export const DEFAULT_TIME_FORMAT = 'HH:mm:ss';

function getTimestamp(): string {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    return `${hours}:${minutes}:${seconds}`;
}

export class Logger {
    static log(message: string): null {
        appendFileSync(DEFAULT_LOG_FILE, `${getTimestamp()} - ${message}\n`);
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
        appendFileSync(DEFAULT_LOG_FILE, `${getTimestamp()} - ILLEGAL -${moveType != MoveType.None? MoveType[moveType] + ',' : ''}, ${reason}\n`);
        return null;
    }
}
