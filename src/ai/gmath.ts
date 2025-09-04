import { GameState } from "../core/states";

export class GMath {
    static clamp(val: number, max: number, min?: number) {
        if(val > max) val = max;
        if(min && val < min) val = min;
        return val;
    }

	static distance(state: GameState, tileIndex1: number, tileIndex2: number, manhattan = false) {
		const cache = distanceCache.get(tileIndex1);
		const size = state.settings.size;

		if (cache) {
			const cachedDistance = cache.get(tileIndex2);
			if (cachedDistance != undefined) return cachedDistance;
		} 
		else {
			distanceCache.set(tileIndex1, new Map());
		}

		const dx = Math.abs((tileIndex1 % size) - (tileIndex2 % size));
		const dy = Math.abs(Math.floor(tileIndex1 / size) - Math.floor(tileIndex2 / size));
		const distance = manhattan ? Math.abs(dx + dy) : Math.max(dx, dy);

		distanceCache.get(tileIndex1)!.set(tileIndex2, distance);

		return distance;
	}
	
	static zeros<T extends number[]>(length: number): T {
		return Array.from({ length }, () => 0) as T;
	}

	static probabilities<T extends number[]>(
		arr: T,
		index: number,
		newValue: number
	): T {
		if (newValue < 0 || newValue > 1) {
			throw new Error("Value must be between 0 and 1");
		}

		const remainingTotal = 1 - newValue;
		const currentRestSum = arr.reduce(
			(sum, val, i) => (i === index ? sum : sum + val),
			0
		);

		return arr.map((val, i, a) => {
			if (i === index) return newValue;

			if (currentRestSum === 0) {
				// spread evenly
				const others = a.length - 1;
				return remainingTotal / others;
			}

			// scale proportionally
			return (val / currentRestSum) * remainingTotal;
		}) as T;
	}

	static setMultiple<T extends number[]>(
		arr: T,
		assignments: Partial<Record<number, number>>
	): T {
		// make a copy
		const out = arr.slice() as number[];

		// apply explicit assignments
		for (const [k, v] of Object.entries(assignments)) {
			const idx = Number(k);
			if (v! < 0 || v! > 1) throw new Error("Value must be between 0 and 1");
			out[idx] = v!;
		}

		// compute remaining total and sum of unassigned current values
		const assignedSum = Object.values(assignments).reduce((s, x) => s! + x!, 0)!;
		const remainingTotal = 1 - assignedSum;

		const unassignedIndices = out
			.map((_, i) => i)
			.filter(i => !(i in assignments));

		const currentUnassignedSum = unassignedIndices.reduce((s, i) => s + (arr[i] ?? 0), 0);

		if (currentUnassignedSum === 0) {
			// split evenly among remaining indices
			const each = remainingTotal / unassignedIndices.length;
			for (const i of unassignedIndices) out[i] = each;
		} else {
			// preserve ratios among the unassigned slots
			for (const i of unassignedIndices) {
				out[i] = (arr[i] / currentUnassignedSum) * remainingTotal;
			}
		}

		return out as T;
	}
}

export let distanceCache = new Map<number, Map<number, number>>();

/**
 * @obsolete Use GMath.distance instead
 * @param tileIndex1 
 * @param tileIndex2 
 * @param size 
 * @param manhattan 
 * @returns 
 */
export function calculateDistance(tileIndex1: number, tileIndex2: number, size: number, manhattan = false) {
	const cache = distanceCache.get(tileIndex1);

	if (cache) {
		const cachedDistance = cache.get(tileIndex2);
		if (cachedDistance != undefined) return cachedDistance;
	} 
	else {
		distanceCache.set(tileIndex1, new Map());
	}

	const dx = Math.abs((tileIndex1 % size) - (tileIndex2 % size));
	const dy = Math.abs(Math.floor(tileIndex1 / size) - Math.floor(tileIndex2 / size));
	const distance = manhattan ? Math.abs(dx + dy) : Math.max(dx, dy);

	distanceCache.get(tileIndex1)!.set(tileIndex2, distance);

	return distance;
}
