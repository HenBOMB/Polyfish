import { TribeType } from "../../core/types";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'fs';

const DATA_DIR = `./data/values`;

if (!existsSync(`${DATA_DIR}`)) {
    mkdirSync(`${DATA_DIR}`);
}

export default class Values<T extends string | number | symbol> {
    private values: Record<T, number>;
    
    constructor(
        readonly tribeType: TribeType,
        readonly name: string
    ) {
        this.values = { } as any;
        this.load();
    }

    get(unitType: T): number {
        const v = this.values[unitType];
        if (!v) {
            throw Error(`No value assigned for ${String(unitType)}`);
        }
        return v;
    }

    set(unitType: T, value: number): void {
        this.values[unitType] = value;
    }

    load(name: string = this.name): void {
        const folder = `${DATA_DIR}/${this.tribeType}`;
        const path = `${folder}/${name}.json`;

        if(!existsSync(path)) {
            // throw new Error(`No such file: ${path}`);
            console.warn(`No load file found, generating from recommendations`);
            try {
                this.recommend();
            }
            catch (e) {
                console.error(e)
                console.warn(`Something went wrong, generating from randomization`);
                this.randomize();
            }
            console.warn(`Saved successfully`);
            this.save();
            return
        }
        this.load(JSON.parse(readFileSync(path, 'utf8')));
    }

    load_values(values: Record<T, number>): void {
        this.values = values;
    }

    save(name: string = this.name): void {
        const folder = `${DATA_DIR}/${this.tribeType}`;

        if(!existsSync(folder)) {
            mkdirSync(folder)
        }

        writeFileSync(`${folder}/${name}.json`, JSON.stringify(this.values));
    }

    recommend(): void {
        throw new Error(`recommend() not implemented`);
    }

    randomize(min: number = 0.01): void {
        const keys = Object.keys(this.values) as T[];
        
        if (!keys.length) {
            throw new Error(`No values to randomize`);
        }

        for (const key of keys) {
            this.values[key] = min + (Math.random() * 2 - 1) * 0.1;
        }
    }
}
