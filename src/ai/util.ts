/**
 * @Obsolete use GMath.dirichlet
 * Sample from a symmetric Dirichlet(α) distribution of given size.
 */
export function sampleDirichlet(alpha: number, size: number): number[] {
    function randGamma(k: number): number {
        if (k < 1) {
            const u = Math.random();
            return randGamma(k + 1) * Math.pow(u, 1 / k);
        }
        const d = k - 1 / 3;
        const c = 1 / Math.sqrt(9 * d);
        while (true) {
            let x: number, v: number;
            do {
                const u1 = Math.random(), u2 = Math.random();
                x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                v = 1 + c * x;
            } while (v <= 0);
            v = v * v * v;
            const u = Math.random();
            if (u < 1 - 0.0331 * x * x * x * x) return d * v;
            if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
        }
    }
  
    const xs = Array.from({ length: size }, () => randGamma(alpha));
    const sum = xs.reduce((a, b) => a + b, 0);
    return xs.map(x => x / sum);
}

export function sampleFromDistribution(dist: number[]): number {
    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < dist.length; i++) {
        cum += dist[i];
        if (r < cum) {
            return i;
        }
    }
    return dist.length - 1;
}