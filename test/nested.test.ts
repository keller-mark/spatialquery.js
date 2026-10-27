import { describe, it, expect, vi } from "vitest";
import { nestedMotifDescent, applyFdrCorrection, type FrequentPattern } from "../src/index.js";

// Frequent patterns (ifMax=false); ABC is the only maximal pattern.
// AB has the same support as its parent ABC, so it is redundant (rule 7).
const frequentPatterns: FrequentPattern[] = [
  { itemsets: ["A"], support: 0.9 },
  { itemsets: ["B"], support: 0.8 },
  { itemsets: ["C"], support: 0.7 },
  { itemsets: ["A", "C"], support: 0.6 },
  { itemsets: ["B", "C"], support: 0.55 },
  { itemsets: ["A", "B"], support: 0.5 },
  { itemsets: ["A", "B", "C"], support: 0.5 },
];

const pValues: Record<string, number> = {
  "A,B,C": 0.5, "A,B": 0.001, "A,C": 0.9, "B,C": 0.8, "A": 0.2, "B": 0.01, "C": 0.3,
};

// Fake enrichment test: looks up a fixed p-value for each motif, then applies FDR correction.
const testMotifs = vi.fn(async (motifs: string[][]) => applyFdrCorrection(
  motifs.map(m => ({ motifs: m.toSorted(), pValue: pValues[m.toSorted().join(",")] })),
));

describe("nestedMotifDescent", () => {
  it("descends into non-significant patterns and drops redundant sub-patterns", async () => {
    testMotifs.mockClear();
    const results = await nestedMotifDescent({ testMotifs, frequentPatterns });

    // Level 0: ABC (not significant). Level 1: AB (significant), AC, BC (not significant).
    // Level 2: the singletons of AC and BC.
    expect(testMotifs.mock.calls.map(([motifs]) => motifs)).toEqual([
      [["A", "B", "C"]],
      [["A", "B"], ["A", "C"], ["B", "C"]],
      [["A"], ["C"], ["B"]],
    ]);
    // AB was tested (and counts toward the correction) but is redundant with ABC.
    expect(results.map(r => r.motifs.join(",")).toSorted()).toEqual(["A", "A,B,C", "A,C", "B", "B,C", "C"]);
    // FDR correction over all 7 tested motifs: B has p = 0.01, rank 2 of 7 -> 0.01 * 7 / 2.
    const b = results.find(r => r.motifs.join(",") === "B")!;
    expect(b.adjPValue).toBeCloseTo(0.035, 12);
    expect(b.ifSignificant).toBe(true);
  });

  it("stops early when the number of tested patterns exceeds the limits", async () => {
    testMotifs.mockClear();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const results = await nestedMotifDescent({ testMotifs, frequentPatterns, maxExpansionFactor: 2 });
    // Only the maximal pattern was tested before the 3 children exceeded 2 x 1 maximal patterns.
    expect(testMotifs).toHaveBeenCalledTimes(1);
    expect(results.map(r => r.motifs)).toEqual([["A", "B", "C"]]);
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });

  it("returns no results when there are no frequent patterns", async () => {
    expect(await nestedMotifDescent({ testMotifs, frequentPatterns: [] })).toEqual([]);
  });
});
