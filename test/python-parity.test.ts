// Compare the outputs of the Python SpatialQuery functions (see fixtures/*.py) to the JS port.
import { describe, it, expect, beforeAll } from "vitest";
import { fileURLToPath } from "url";
import { resolve, dirname, join } from "path";
import { readFileSync } from "fs";

import FileSystemStore from '@zarrita/storage/fs';
import { HdfStore } from "hdf5-as-virtual-zarr";
import { readZarr, get } from 'anndata.js';
import { assert } from './assert.js';
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
  type SqContext,
  type FrequentPattern,
  type MotifEnrichmentResult,
} from '../src/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const fixturesDir = resolve(__dirname, "..", "fixtures");

const gridFixture = JSON.parse(readFileSync(join(fixturesDir, "secondary_analysis.find_patterns_grid.json"), "utf-8"));
const motifFixture = JSON.parse(readFileSync(join(fixturesDir, "secondary_analysis.motif_fixtures.json"), "utf-8"));

type PyPattern = { support: number, itemsets: string[], neighbor_id?: number[] };
type PyEnrichment = {
  center: string, motifs: string[], n_center_motif: number, n_center: number, n_motif: number,
  expectation: number, 'p-values': number, 'adj-pval'?: number, if_significant: boolean,
  neighbor_id?: number[], center_id?: number[],
};

async function loadContext(): Promise<SqContext> {
  const internalStore = new FileSystemStore(join(fixturesDir, "secondary_analysis.new.h5ad"));
  const store = await HdfStore.fromStore(internalStore);
  const adata = await readZarr(store);

  const obsIndex = Array.from((await get(await adata.obsNames(), [null])).data as ArrayLike<string>);

  const predictedLabel = await adata.obs.get("predicted_label");
  assert("codes" in predictedLabel, "Was not categorical");
  const labelCodes = await get(predictedLabel.codes, [null]);
  const labelCats = await get(predictedLabel.categories, [null]);
  const labels = Array.from(labelCodes.data as ArrayLike<number>).map(code => (labelCats.data as ArrayLike<string>)[code]);
  const obsIdToLabel = new Map(obsIndex.map((obsId, i) => [obsId, labels[i]]));

  const spatialCoordsArr = await adata.obsm.get("X_spatial");
  assert(!("codes" in spatialCoordsArr), "Was unexpectedly categorical");
  assert(!("indptr" in spatialCoordsArr), "Was unexpectedly sparse");
  assert(!("axisRoot" in spatialCoordsArr), "Was unexpectedly axis_arrays");
  const spatialX = await get(spatialCoordsArr, [null, 0]);
  const spatialY = await get(spatialCoordsArr, [null, 1]);

  return buildContext({
    obsIndex,
    xCoords: Float64Array.from(spatialX.data as ArrayLike<number>),
    yCoords: Float64Array.from(spatialY.data as ArrayLike<number>),
    obsIdToObsSetName: (obsId: string) => obsIdToLabel.get(obsId)!,
    normalizeSpatialCoord: false,
  });
}

const key = (motif: string[]) => JSON.stringify(motif.toSorted());

// Compare frequent patterns irrespective of the order of patterns with tied support.
function expectPatternsToMatch(js: FrequentPattern[], py: PyPattern[], obsIndex: string[]) {
  expect(js.map(p => key(p.itemsets)).toSorted()).toEqual(py.map(p => key(p.itemsets)).toSorted());
  const jsByKey = new Map(js.map(p => [key(p.itemsets), p]));
  for (const pyPattern of py) {
    const jsPattern = jsByKey.get(key(pyPattern.itemsets))!;
    expect(jsPattern.support).toBeCloseTo(pyPattern.support, 12);
    if (pyPattern.neighbor_id) {
      expect(jsPattern.neighborIds!.toSorted()).toEqual(pyPattern.neighbor_id.map(i => obsIndex[i]).toSorted());
    }
  }
  // Sorted by descending support.
  js.slice(1).forEach((p, i) => expect(p.support).toBeLessThanOrEqual(js[i].support));
}

function expectEnrichmentToMatch(js: MotifEnrichmentResult[], py: PyEnrichment[], obsIndex: string[]) {
  expect(js.map(r => key(r.motifs)).toSorted()).toEqual(py.map(r => key(r.motifs)).toSorted());
  const jsByKey = new Map(js.map(r => [key(r.motifs), r]));
  for (const pyResult of py) {
    const jsResult = jsByKey.get(key(pyResult.motifs))!;
    expect(jsResult.center).toBe(pyResult.center);
    expect(jsResult.motifs).toEqual(pyResult.motifs);
    expect(jsResult.nCenterMotif).toBe(pyResult.n_center_motif);
    expect(jsResult.nCenter).toBe(pyResult.n_center);
    expect(jsResult.nMotif).toBe(pyResult.n_motif);
    expect(jsResult.expectation).toBeCloseTo(pyResult.expectation, 9);
    expectRelativelyClose(jsResult.pValue, pyResult['p-values']);
    if (pyResult['adj-pval'] !== undefined) {
      expectRelativelyClose(jsResult.adjPValue!, pyResult['adj-pval']);
    } else {
      expect(jsResult.adjPValue).toBeUndefined();
    }
    expect(jsResult.ifSignificant).toBe(pyResult.if_significant);
    if (pyResult.center_id) {
      expect(jsResult.centerIds!.toSorted()).toEqual(pyResult.center_id.map(i => obsIndex[i]).toSorted());
      expect(jsResult.neighborIds!.toSorted()).toEqual(pyResult.neighbor_id!.map(i => obsIndex[i]).toSorted());
    }
  }
  // Sorted by ascending adjusted p-value when there are multiple motifs.
  if (js.length > 1) {
    js.slice(1).forEach((r, i) => expect(r.adjPValue!).toBeGreaterThanOrEqual(js[i].adjPValue!));
  }
}

function expectRelativelyClose(actual: number, expected: number, relTol = 1e-8) {
  if (expected === 0) {
    expect(actual).toBeLessThan(1e-300);
  } else {
    expect(Math.abs(actual - expected) / Math.abs(expected)).toBeLessThan(relTol);
  }
}

describe("Compare Python outputs to JS outputs", () => {
  let context: SqContext;
  const params = motifFixture.params;

  beforeAll(async () => {
    context = await loadContext();
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
    expectPatternsToMatch(patterns, motifFixture.find_fp_knn, context.obsIndex);
  });

  it("findFpDist", async () => {
    const patterns = await findFpDist(context, {
      ct: params.ct, maxDist: params.max_dist, minSize: params.min_size, minSupport: params.min_support,
    });
    expectPatternsToMatch(patterns, motifFixture.find_fp_dist, context.obsIndex);
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

  it("motif enrichment rejects an unknown center cell type", async () => {
    await expect(motifEnrichmentDist(context, { ct: "not a cell type" })).rejects.toThrow("Found no");
    await expect(motifEnrichmentKnn(context, { ct: "not a cell type" })).rejects.toThrow("Found no");
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
      const pyByKey = new Map(pyPatterns.map(p => [key(p.itemsets), p]));
      for (const p of patterns) {
        const pyPattern = pyByKey.get(key(p.itemsets));
        if (pyPattern) {
          expect(Math.abs(p.support - pyPattern.support)).toBeLessThan(tolerance);
        }
      }
    });
  });
});
