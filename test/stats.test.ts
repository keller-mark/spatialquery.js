import { describe, it, expect } from "vitest";
import { hypergeom, fdrCorrection, roundHalfEven, normSf, mannWhitneyU, rankAverage } from "../src/stats.js";
import { computeMotifEnrichment, applyFdrCorrection } from "../src/index.js";

// Expected values generated with scipy.stats.hypergeom(M=M, n=n, N=N).
const hypergeomCases: [number, number, number, number, number, number][] = [
  // M, n, N, k, mean, sf(k)
  [4058, 500, 1200, 150, 147.85608674223755, 0.38911749946783847],
  [4058, 500, 1200, 200, 147.85608674223755, 3.681080796606616e-08],
  [4058, 500, 1200, 100, 147.85608674223755, 0.9999998417414943],
  [20, 7, 12, 3, 4.2, 0.7492260061919505],
  [100000, 3000, 5000, 400, 150.0, 2.0382808365550979e-72],
  [4058, 500, 1200, 500, 147.85608674223755, 0.0],
  [10, 3, 4, 0, 1.2, 0.8333333333333334],
];

describe("hypergeom", () => {
  it.each(hypergeomCases)("M=%i n=%i N=%i k=%i matches scipy", (M, n, N, k, mean, sf) => {
    const h = hypergeom(M, n, N);
    expect(h.mean()).toBeCloseTo(mean, 10);
    if (sf === 0) {
      expect(h.sf(k)).toBe(0);
    } else {
      expect(Math.abs(h.sf(k) - sf) / sf).toBeLessThan(1e-9);
    }
  });
});

describe("fdrCorrection", () => {
  it("matches statsmodels fdrcorrection(method='poscorr')", () => {
    const { rejected, adjusted } = fdrCorrection([0.01, 0.04, 0.03, 0.2, 0.001], 0.05);
    expect(rejected).toEqual([true, true, true, false, true]);
    [0.025, 0.05, 0.05, 0.2, 0.005].forEach((p, i) => expect(adjusted[i]).toBeCloseTo(p, 12));
  });
});

describe("normSf", () => {
  // Expected values generated with scipy.stats.norm.sf(z).
  it.each([
    [-2.0, 0.9772498680518208],
    [0.0, 0.5],
    [0.5, 0.3085375387259869],
    [1.96, 0.024997895148220435],
    [2.9, 0.0018658133003840375],
    [3.1, 0.0009676032132183563],
    [5.0, 2.8665157187919333e-07],
    [10.0, 7.61985302416047e-24],
    [30.0, 4.906713927147908e-198],
  ])("normSf(%d) matches scipy", (z, expected) => {
    expect(Math.abs(normSf(z) - expected) / expected).toBeLessThan(1e-10);
  });
});

describe("mannWhitneyU", () => {
  // Expected values generated with scipy.stats.mannwhitneyu(x, y, alternative='two-sided', method='auto').
  const linspace = (start: number, stop: number, n: number) => Array.from({ length: n }, (_, i) => start + (stop - start) * i / (n - 1));
  it.each([
    ["exact", [0.1, 0.4, 0.35, 0.8], [0.2, 0.05, 0.15], 0.22857142857142856],
    ["exact (n=8)", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.97], [0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.65, 0.75], 0.004662004662004662],
    ["exact (one large sample)", [1, 2, 3], linspace(4, 13, 10), 0.006993006993006993],
    ["asymptotic (ties)", [0.1, 0.1, 0.4, 0.5], [0.2, 0.1, 0.0], 0.3590120537864294],
    ["asymptotic (n>8)", linspace(0, 1, 12), linspace(0.3, 1.5, 10), 0.0321141793079075],
  ])("%s matches scipy", (_, x, y, expected) => {
    expect(Math.abs(mannWhitneyU(x as number[], y as number[]).pValue - (expected as number)) / (expected as number)).toBeLessThan(1e-10);
  });

  it("returns NaN when all values are tied, as scipy does", () => {
    expect(mannWhitneyU([0, 0, 0], [0, 0]).pValue).toBeNaN();
  });
});

describe("rankAverage", () => {
  it("assigns tied values their average rank", () => {
    expect(rankAverage([0.3, 0.1, 0.3, 0.2, 0.3])).toEqual([4, 1, 4, 2, 4]);
  });
});

describe("roundHalfEven", () => {
  it("matches Python round()", () => {
    expect([0.5, 1.5, 2.5, 3.5, 2.4, 2.6].map(roundHalfEven)).toEqual([0, 2, 2, 4, 2, 3]);
  });
});

describe("computeMotifEnrichment", () => {
  it("computes the hypergeometric test for a motif", () => {
    const result = computeMotifEnrichment({
      center: "A", motif: ["C", "B"], numCells: 4058, numCenter: 500, numMotif: 1200, numCenterMotif: 200,
    });
    expect(result.motifs).toEqual(["B", "C"]);
    expect(result.nCenter).toBe(500);
    expect(result.expectation).toBeCloseTo(147.85608674223755, 10);
    expect(result.pValue).toBeCloseTo(3.681080796606616e-08, 15);
  });

  it("divides the center count when the center cell type is in the motif", () => {
    const result = computeMotifEnrichment({
      center: "A", motif: ["A", "B", "A"], numCells: 100, numCenter: 5, numMotif: 10, numCenterMotif: 1,
    });
    // round(5 / 2) == 2 in Python
    expect(result.nCenter).toBe(2);
  });

  it("applies FDR correction across motifs", () => {
    const base = { center: "A", motifs: ["B"], nCenterMotif: 0, nCenter: 0, nMotif: 0, expectation: 0 };
    const single = applyFdrCorrection([{ ...base, pValue: 0.04 }]);
    expect(single[0].ifSignificant).toBe(true);
    expect(single[0].adjPValue).toBeUndefined();

    const multi = applyFdrCorrection([0.01, 0.04, 0.03, 0.2, 0.001].map(pValue => ({ ...base, pValue })));
    // Sorted by adjusted p-value (0.03 and 0.04 are tied at an adjusted p-value of 0.05).
    expect(multi.map(r => r.pValue).slice(0, 2)).toEqual([0.001, 0.01]);
    expect(multi[4].pValue).toBe(0.2);
    expect(multi.map(r => r.ifSignificant)).toEqual([true, true, true, true, false]);
  });
});
