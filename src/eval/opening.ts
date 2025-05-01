export class Opening {
    private moves: string[];
    private currentIndex: number;

    constructor(moves: string[]) {
        this.moves = moves;
        this.currentIndex = 0;
    }

    public nextMove(): string | null {
        if (this.currentIndex < this.moves.length) {
            return this.moves[this.currentIndex++];
        }
        return null; // No more moves
    }

    public reset(): void {
        this.currentIndex = 0;
    }

    public hasNext(): boolean {
        return this.currentIndex < this.moves.length;
    }
}

// Example usage:
const opening = new Opening(["e4", "e5", "Nf3", "Nc6"]);
console.log(opening.nextMove()); // "e4"
console.log(opening.nextMove()); // "e5"
console.log(opening.hasNext()); // true
opening.reset();
console.log(opening.nextMove()); // "e4"

// move unit to where leads to more discovery?