import { describe, it, expect } from "vitest";
import { hypergeom, fdrCorrection, roundHalfEven } from "../src/stats.js";
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
