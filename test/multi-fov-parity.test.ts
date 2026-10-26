// Compare the outputs of the Python multi-FOV SpatialQuery functions
// (see fixtures/generate_multi_fov_fixtures.py) to the JS port.
import { describe, it, expect, beforeAll } from "vitest";
import {
  buildMultiContext,
  multiFindFpKnn,
  multiFindFpDist,
  findFpKnnFov,
  findFpDistFov,
  multiMotifEnrichmentKnn,
  multiMotifEnrichmentDist,
  multiMotifCompositionKnn,
  multiMotifCompositionDist,
  differentialAnalysisKnn,
  differentialAnalysisDist,
  differentialTestsKnn,
  differentialTestsDist,
  type SqMultiContext,
  type FovInitParams,
  type MultiFovMotifEnrichmentResult,
  type DifferentialResult,
  type DifferentialTestResult,
} from '../src/index.js';
import {
  readFixture,
  loadCellData,
  toObsSetNameFunc,
  motifKey,
  expectRelativelyClose,
  expectPatternsToMatch,
  expectEnrichmentStatsToMatch,
  expectCompositionToMatch,
  type PyEnrichment,
} from './fixture-data.js';

const fixture = readFixture("secondary_analysis.multi_fov_fixtures.json");
const params = fixture.params;

// Python's str(sorted(motif)), used as keys of the cell ID dicts.
const pythonMotifStr = (motif: string[]) => `[${motif.toSorted().map(m => `'${m}'`).join(", ")}]`;

type PyCellIds = Record<string, Record<string, string[]>>;

function expectEnrichmentToMatch(
  js: MultiFovMotifEnrichmentResult[],
  py: PyEnrichment[],
  pyMotifIds?: PyCellIds,
  pyCenterIds?: PyCellIds,
) {
  const jsByKey = expectEnrichmentStatsToMatch(js, py);
  if (pyMotifIds && pyCenterIds) {
    for (const pyResult of py) {
      const jsResult = jsByKey.get(motifKey(pyResult.motifs))!;
      const sortValues = (ids: Record<string, string[]>) => Object.fromEntries(
        Object.entries(ids).map(([ds, cellIds]) => [ds, cellIds.toSorted()]),
      );
      expect(sortValues(jsResult.neighborIds!)).toEqual(pyMotifIds[pythonMotifStr(pyResult.motifs)]);
      expect(sortValues(jsResult.centerIds!)).toEqual(pyCenterIds[pythonMotifStr(pyResult.motifs)]);
    }
  }
}

type PyDifferential = {
  significant: Record<string, Record<string, unknown>[]>,
  tests: { itemsets: string[], support_0: number[], support_1: number[], p_value: number }[],
};

function expectDifferentialToMatch(
  tests: DifferentialTestResult[],
  significant: Record<string, DifferentialResult[]>,
  py: PyDifferential,
) {
  const [d0, d1] = params.datasets;

  // Every motif tested, with its per-FOV supports and Mann-Whitney U p-value.
  expect(tests.map(t => motifKey(t.itemsets)).toSorted()).toEqual(py.tests.map(t => motifKey(t.itemsets)).toSorted());
  const testsByKey = new Map(tests.map(t => [motifKey(t.itemsets), t]));
  for (const pyTest of py.tests) {
    const test = testsByKey.get(motifKey(pyTest.itemsets))!;
    expect(test.supports[d0].length).toBe(pyTest.support_0.length);
    test.supports[d0].forEach((s, i) => expect(s).toBeCloseTo(pyTest.support_0[i], 12));
    test.supports[d1].forEach((s, i) => expect(s).toBeCloseTo(pyTest.support_1[i], 12));
    expectRelativelyClose(test.pValue, pyTest.p_value);
  }

  // The significant motifs, split by the dataset with the higher frequency.
  expect(Object.keys(significant).toSorted()).toEqual([d0, d1].toSorted());
  for (const d of [d0, d1]) {
    const pyRows = py.significant[d];
    expect(significant[d].map(r => motifKey(r.itemsets)).toSorted())
      .toEqual(pyRows.map(r => motifKey(r.itemsets as string[])).toSorted());
    const byKey = new Map(significant[d].map(r => [motifKey(r.itemsets), r]));
    for (const pyRow of pyRows) {
      const row = byKey.get(motifKey(pyRow.itemsets as string[]))!;
      expect(row.supportMean[d0]).toBeCloseTo(pyRow[`support_${d0}_mean`] as number, 12);
      expect(row.supportMean[d1]).toBeCloseTo(pyRow[`support_${d1}_mean`] as number, 12);
      expectRelativelyClose(row.adjPValue, pyRow['adj-pval'] as number);
    }
  }
}

describe("Compare Python multi-FOV outputs to JS outputs", () => {
  let multi: SqMultiContext;

  beforeAll(async () => {
    const { obsIndex, labels, xCoords, yCoords } = await loadCellData();
    const cellFov: number[] = fixture.cell_fov;
    // Python derives unique FOV names from the (repeated) dataset names, e.g. 'kidney_left' -> 'kidney-left_0'.
    // The Python outputs are keyed by these names, so we use them as the explicit dataset IDs,
    // and the corresponding dataset names (e.g. 'kidney-left') as the dataset classes.
    const fovs: FovInitParams[] = fixture.renamed_datasets.map((datasetId: string, fov: number) => {
      const indices = cellFov.flatMap((f, i) => (f === fov ? [i] : []));
      const fovObsIndex = indices.map(i => obsIndex[i]);
      return {
        datasetId,
        datasetClass: datasetId.split('_')[0],
        obsIndex: fovObsIndex,
        xCoords: Float64Array.from(indices, i => xCoords[i]),
        yCoords: Float64Array.from(indices, i => yCoords[i]),
        obsIdToObsSetName: toObsSetNameFunc(fovObsIndex, indices.map(i => labels[i])),
        normalizeSpatialCoord: false,
      };
    });
    multi = buildMultiContext(fovs);
  });

  it("buildMultiContext keeps the dataset IDs and classes, and rejects duplicate IDs", () => {
    expect(multi.datasetIds).toEqual(fixture.renamed_datasets);
    expect(new Set(multi.datasetClasses)).toEqual(new Set(params.datasets));
    const fov = { obsIndex: ["a", "b"], xCoords: Float64Array.of(0, 1), yCoords: Float64Array.of(0, 1), obsIdToObsSetName: () => "T cell" };
    expect(() => buildMultiContext([
      { ...fov, datasetId: "sample_1", datasetClass: "control" },
      { ...fov, datasetId: "sample_1", datasetClass: "disease" },
    ])).toThrow("Dataset IDs must be unique");
  });

  it("multiFindFpKnn", async () => {
    const patterns = await multiFindFpKnn(multi, {
      ct: params.ct, k: params.k, minSupport: params.min_support, maxDist: params.max_dist,
    });
    expectPatternsToMatch(patterns, fixture.find_fp_knn);
  });

  it("multiFindFpDist with a dataset class filter", async () => {
    const patterns = await multiFindFpDist(multi, {
      ct: params.ct, datasetClass: "kidney-left", maxDist: params.max_dist, minSize: params.min_size, minSupport: params.min_support,
    });
    expectPatternsToMatch(patterns, fixture.find_fp_dist_left);
  });

  it("findFpKnnFov / findFpDistFov", async () => {
    expectPatternsToMatch(await findFpKnnFov(multi, {
      ct: params.ct, datasetId: "kidney-left_0", k: params.k, minSupport: params.min_support, maxDist: params.max_dist,
    }), fixture.find_fp_knn_fov);
    expectPatternsToMatch(await findFpDistFov(multi, {
      ct: params.ct, datasetId: "kidney-right_3", maxDist: params.max_dist, minSize: params.min_size, minSupport: params.min_support,
    }), fixture.find_fp_dist_fov);
    await expect(findFpKnnFov(multi, { ct: params.ct, datasetId: "kidney-left_99" })).rejects.toThrow("Found no");
  });

  it("multiMotifEnrichmentKnn", async () => {
    const results = await multiMotifEnrichmentKnn(multi, {
      ct: params.ct, k: params.k, minSupport: params.min_support, maxDist: params.max_dist, returnCellIds: true,
    });
    expectEnrichmentToMatch(results, fixture.motif_enrichment_knn,
      fixture.motif_enrichment_knn_motif_ids, fixture.motif_enrichment_knn_center_ids);
  });

  it("multiMotifEnrichmentDist", async () => {
    const results = await multiMotifEnrichmentDist(multi, {
      ct: params.ct, maxDist: params.max_dist, minSize: params.min_size, minSupport: params.min_support, returnCellIds: true,
    });
    expectEnrichmentToMatch(results, fixture.motif_enrichment_dist,
      fixture.motif_enrichment_dist_motif_ids, fixture.motif_enrichment_dist_center_ids);
  });

  it("multiMotifEnrichmentKnn with specified motifs and a dataset class filter", async () => {
    const results = await multiMotifEnrichmentKnn(multi, {
      ct: params.ct, motifs: params.specified_motifs, datasetClass: "kidney-right", k: params.k, maxDist: params.max_dist,
    });
    expectEnrichmentToMatch(results, fixture.motif_enrichment_knn_specified_right);
  });

  it("multiMotifEnrichmentDist with mode='nested'", async () => {
    const results = await multiMotifEnrichmentDist(multi, {
      ct: params.ct, maxDist: params.max_dist, minSize: params.min_size, minSupport: params.min_support, mode: 'nested',
    });
    expect(results.length).toBeGreaterThan(fixture.motif_enrichment_dist.length);
    expectEnrichmentToMatch(results, fixture.motif_enrichment_dist_nested);
  });

  it("multi-FOV functions reject invalid inputs", async () => {
    await expect(multiMotifEnrichmentDist(multi, { ct: "not a cell type" })).rejects.toThrow("Found no");
    await expect(multiFindFpKnn(multi, { ct: params.ct, datasetClass: "not a class" })).rejects.toThrow("Invalid dataset class");
    await expect(multiMotifEnrichmentKnn(multi, { ct: params.ct, motifs: ["not a cell type"] })).rejects.toThrow("All cell types in motifs");
  });

  it("multiMotifCompositionKnn", () => {
    const rows = multiMotifCompositionKnn(multi, {
      ct: params.ct, motifs: params.specified_motifs, k: params.k, maxDist: params.max_dist,
    });
    expectCompositionToMatch(rows, fixture.motif_composition_knn);
  });

  it("multiMotifCompositionDist with a dataset class filter", () => {
    const rows = multiMotifCompositionDist(multi, {
      ct: params.ct, motifs: params.specified_motifs, datasetClass: "kidney-left", maxDist: params.max_dist,
    });
    expectCompositionToMatch(rows, fixture.motif_composition_dist_left);
  });

  it("differentialAnalysisKnn", async () => {
    const diffParams = {
      ct: params.ct, datasetClasses: params.datasets, k: params.k, minSupport: params.min_support, maxDist: params.max_dist,
    };
    const tests = await differentialTestsKnn(multi, diffParams);
    const significant = await differentialAnalysisKnn(multi, diffParams);
    expectDifferentialToMatch(tests, significant, fixture.differential_analysis_knn);
  });

  it("differentialAnalysisDist", async () => {
    const diffParams = {
      ct: params.ct, datasetClasses: params.datasets, maxDist: params.max_dist, minSupport: params.min_support, minSize: params.min_size,
    };
    const tests = await differentialTestsDist(multi, diffParams);
    const significant = await differentialAnalysisDist(multi, diffParams);
    expectDifferentialToMatch(tests, significant, fixture.differential_analysis_dist);
  });

  it("differentialAnalysisDist with specified motifs", async () => {
    const diffParams = {
      ct: params.ct, datasetClasses: params.datasets, motifs: params.specified_motifs, maxDist: params.max_dist,
    };
    const tests = await differentialTestsDist(multi, diffParams);
    const significant = await differentialAnalysisDist(multi, diffParams);
    // Both specified motifs are significantly more frequent in one of the dataset classes.
    expect(Object.values(significant).flat()).toHaveLength(2);
    expectDifferentialToMatch(tests, significant, fixture.differential_analysis_dist_specified);
    await expect(differentialAnalysisDist(multi, { ...diffParams, datasetClasses: ["kidney-left"] })).rejects.toThrow("Require 2 dataset classes");
  });
});
