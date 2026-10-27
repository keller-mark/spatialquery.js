// Compare the outputs of the Python SpatialQuery functions (see fixtures/*.py) to the JS port.
import { describe, it, expect, beforeAll } from "vitest";
import {
  buildContext,
  buildGridPoints,
  findPatternsGrid,
  findPatternsRand,
  findFpKnn,
  findFpDist,
  findMaximalPatterns,
  motifEnrichmentKnn,
  motifEnrichmentDist,
  motifCompositionKnn,
  motifCompositionDist,
  type SqContext,
  type FrequentPattern,
  type MotifEnrichmentResult,
} from '../src/index.js';
import {
  readFixture,
  loadCellData,
  toObsSetNameFunc,
  motifKey,
  expectPatternsToMatch,
  expectEnrichmentStatsToMatch,
  expectCompositionToMatch,
  type PyPattern,
  type PyEnrichment,
} from './fixture-data.js';

const gridFixture = readFixture("secondary_analysis.find_patterns_grid.json");
const motifFixture = readFixture("secondary_analysis.motif_fixtures.json");

function expectEnrichmentToMatch(js: MotifEnrichmentResult[], py: PyEnrichment[], obsIndex: string[]) {
  const jsByKey = expectEnrichmentStatsToMatch(js, py);
  for (const pyResult of py) {
    if (pyResult.center_id) {
      const jsResult = jsByKey.get(motifKey(pyResult.motifs))!;
      expect(jsResult.centerIds!.toSorted()).toEqual(pyResult.center_id.map(i => obsIndex[i]).toSorted());
      expect(jsResult.neighborIds!.toSorted()).toEqual(pyResult.neighbor_id!.map(i => obsIndex[i]).toSorted());
    }
  }
}

describe("Compare Python outputs to JS outputs", () => {
  let context: SqContext;
  const params = motifFixture.params;

  beforeAll(async () => {
    const { obsIndex, labels, xCoords, yCoords } = await loadCellData();
    context = buildContext({
      obsIndex,
      xCoords,
      yCoords,
      obsIdToObsSetName: toObsSetNameFunc(obsIndex, labels),
      normalizeSpatialCoord: false,
    });
    expect(context.obsIndex.length).toEqual(4058);
  });

  it("buildGridPoints", () => {
    const gridPoints = buildGridPoints(context, gridFixture.params.max_dist);
    expect(gridPoints.length).toBe(gridFixture.grid_pos.length);
    gridPoints.forEach(([x, y], i) => {
      expect(x).toBeCloseTo(gridFixture.grid_pos[i][0], 6);
      expect(y).toBeCloseTo(gridFixture.grid_pos[i][1], 6);
    });
  });

  it("findPatternsGrid", async () => {
    const patterns = await findPatternsGrid(context, {
      maxDist: gridFixture.params.max_dist,
      minSupport: gridFixture.params.min_support,
      returnCellIds: true,
    });
    expectPatternsToMatch(patterns, gridFixture.fp_tree, context.obsIndex);
  });

  it("findFpKnn", async () => {
    const patterns = await findFpKnn(context, {
      ct: params.ct, k: params.k, minSupport: params.min_support, maxDist: params.max_dist,
    });
    expectPatternsToMatch(patterns, motifFixture.find_fp_knn);
  });

  it("findFpDist", async () => {
    const patterns = await findFpDist(context, {
      ct: params.ct, maxDist: params.max_dist, minSize: params.min_size, minSupport: params.min_support,
    });
    expectPatternsToMatch(patterns, motifFixture.find_fp_dist);
  });

  it("motifEnrichmentKnn", async () => {
    const results = await motifEnrichmentKnn(context, {
      ct: params.ct, k: params.k, minSupport: params.min_support, maxDist: params.max_dist, returnCellIds: true,
    });
    expectEnrichmentToMatch(results, motifFixture.motif_enrichment_knn, context.obsIndex);
  });

  it("motifEnrichmentDist", async () => {
    const results = await motifEnrichmentDist(context, {
      ct: params.ct, maxDist: params.max_dist, minSize: params.min_size, minSupport: params.min_support, returnCellIds: true,
    });
    expectEnrichmentToMatch(results, motifFixture.motif_enrichment_dist, context.obsIndex);
  });

  it("motifEnrichmentKnn with specified motifs", async () => {
    const results = await motifEnrichmentKnn(context, {
      ct: params.ct, motifs: params.specified_motifs, k: params.k, maxDist: params.max_dist,
    });
    expectEnrichmentToMatch(results, motifFixture.motif_enrichment_knn_specified, context.obsIndex);
  });

  it("motifEnrichmentDist with specified motifs", async () => {
    const results = await motifEnrichmentDist(context, {
      ct: params.ct, motifs: params.specified_motifs, maxDist: params.max_dist,
    });
    expectEnrichmentToMatch(results, motifFixture.motif_enrichment_dist_specified, context.obsIndex);
  });

  it("motifEnrichmentKnn with mode='nested'", async () => {
    const results = await motifEnrichmentKnn(context, {
      ct: params.ct, k: params.k, minSupport: params.min_support, maxDist: params.max_dist, mode: 'nested',
    });
    // The nested descent should test more motifs than the maximal mode.
    expect(results.length).toBeGreaterThan(motifFixture.motif_enrichment_knn.length);
    expectEnrichmentToMatch(results, motifFixture.motif_enrichment_knn_nested, context.obsIndex);
  });

  it("motifEnrichmentDist with mode='nested'", async () => {
    const results = await motifEnrichmentDist(context, {
      ct: params.ct, maxDist: params.max_dist, minSize: params.min_size, minSupport: params.min_support, mode: 'nested',
    });
    expect(results.length).toBeGreaterThan(motifFixture.motif_enrichment_dist.length);
    expectEnrichmentToMatch(results, motifFixture.motif_enrichment_dist_nested, context.obsIndex);
  });

  it("motif enrichment rejects an unknown center cell type or mode", async () => {
    await expect(motifEnrichmentDist(context, { ct: "not a cell type" })).rejects.toThrow("Found no");
    await expect(motifEnrichmentKnn(context, { ct: "not a cell type" })).rejects.toThrow("Found no");
    // @ts-expect-error Testing an invalid mode.
    await expect(motifEnrichmentKnn(context, { ct: params.ct, mode: 'other' })).rejects.toThrow("mode must be");
  });

  it("motifCompositionKnn", () => {
    const rows = motifCompositionKnn(context, {
      ct: params.ct, motifs: params.composition_motifs, k: params.k, maxDist: params.max_dist,
    });
    expectCompositionToMatch(rows, motifFixture.motif_composition_knn);
  });

  it("motifCompositionDist", () => {
    const rows = motifCompositionDist(context, {
      ct: params.ct, motifs: params.composition_motifs, maxDist: params.max_dist,
    });
    expectCompositionToMatch(rows, motifFixture.motif_composition_dist);
  });

  // findPatternsRand uses Math.random (Python uses a seeded numpy RNG),
  // so we check properties of the output and loose agreement with Python.
  describe("findPatternsRand", () => {
    let patterns: FrequentPattern[];
    const nPoints = 5000;

    beforeAll(async () => {
      patterns = await findPatternsRand(context, {
        maxDist: params.max_dist, nPoints, minSupport: params.min_support, minSize: params.min_size, returnCellIds: true,
      });
    });

    it("returns maximal patterns above the minimum support, sorted by support", () => {
      expect(patterns.length).toBeGreaterThan(0);
      expect(findMaximalPatterns(patterns)).toHaveLength(patterns.length);
      for (const p of patterns) {
        expect(p.support).toBeGreaterThanOrEqual(params.min_support);
        expect(p.support).toBeLessThanOrEqual(1);
      }
      patterns.slice(1).forEach((p, i) => expect(p.support).toBeLessThanOrEqual(patterns[i].support));
    });

    it("returns neighbor cells whose cell types belong to the pattern", () => {
      const obsIdToLabel = new Map(context.obsIndex.map((obsId, i) => [obsId, context.labels[i]]));
      for (const p of patterns) {
        expect(p.neighborIds!.length).toBeGreaterThan(0);
        const motifTypes = new Set(p.itemsets);
        const neighborTypes = new Set(p.neighborIds!.map(id => obsIdToLabel.get(id)!));
        // Every neighbor has a motif cell type, and every motif cell type is represented.
        expect(neighborTypes).toEqual(motifTypes);
      }
    });

    it("finds patterns with support similar to Python", () => {
      // With 1000 (Python) and 5000 (JS) random points, support estimates have a standard error
      // of roughly 0.015 and 0.007, so 0.08 is a generous tolerance.
      const tolerance = 0.08;
      const pyPatterns: PyPattern[] = motifFixture.find_patterns_rand;
      for (const pyPattern of pyPatterns.filter(p => p.support >= params.min_support + tolerance)) {
        // Strongly-supported Python patterns should be found by JS too, either directly
        // or as a subset of a (maximal) JS pattern.
        const containing = patterns.filter(p => pyPattern.itemsets.every(ct => p.itemsets.includes(ct)));
        expect(containing.length).toBeGreaterThan(0);
      }
      const pyByKey = new Map(pyPatterns.map(p => [motifKey(p.itemsets), p]));
      for (const p of patterns) {
        const pyPattern = pyByKey.get(motifKey(p.itemsets));
        if (pyPattern) {
          expect(Math.abs(p.support - pyPattern.support)).toBeLessThan(tolerance);
        }
      }
    });
  });
});
