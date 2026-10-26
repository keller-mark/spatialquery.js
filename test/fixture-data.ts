// Helpers for loading the HuBMAP test dataset and the Python reference outputs (see fixtures/).
import { fileURLToPath } from "url";
import { resolve, dirname, join } from "path";
import { readFileSync } from "fs";
import { expect } from "vitest";

import FileSystemStore from '@zarrita/storage/fs';
import { HdfStore } from "hdf5-as-virtual-zarr";
import { readZarr, get } from 'anndata.js';
import { assert } from './assert.js';
import type { FrequentPattern, MotifEnrichmentResult, MotifCompositionRow } from '../src/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
export const fixturesDir = resolve(__dirname, "..", "fixtures");

export function readFixture(fileName: string) {
  return JSON.parse(readFileSync(join(fixturesDir, fileName), "utf-8"));
}

export type CellData = {
  obsIndex: string[],
  labels: string[],
  xCoords: Float64Array,
  yCoords: Float64Array,
};

// Load the obs index, cell type labels, and spatial coordinates of the test dataset.
export async function loadCellData(): Promise<CellData> {
  const internalStore = new FileSystemStore(join(fixturesDir, "secondary_analysis.new.h5ad"));
  const store = await HdfStore.fromStore(internalStore);
  const adata = await readZarr(store);

  const obsIndex = Array.from((await get(await adata.obsNames(), [null])).data as ArrayLike<string>);

  const predictedLabel = await adata.obs.get("predicted_label");
  assert("codes" in predictedLabel, "Was not categorical");
  const labelCodes = await get(predictedLabel.codes, [null]);
  const labelCats = await get(predictedLabel.categories, [null]);
  const labels = Array.from(labelCodes.data as ArrayLike<number>).map(code => (labelCats.data as ArrayLike<string>)[code]);

  const spatialCoordsArr = await adata.obsm.get("X_spatial");
  assert(!("codes" in spatialCoordsArr), "Was unexpectedly categorical");
  assert(!("indptr" in spatialCoordsArr), "Was unexpectedly sparse");
  assert(!("axisRoot" in spatialCoordsArr), "Was unexpectedly axis_arrays");
  const spatialX = await get(spatialCoordsArr, [null, 0]);
  const spatialY = await get(spatialCoordsArr, [null, 1]);

  return {
    obsIndex,
    labels,
    xCoords: Float64Array.from(spatialX.data as ArrayLike<number>),
    yCoords: Float64Array.from(spatialY.data as ArrayLike<number>),
  };
}

// Function mapping obs IDs to cell type labels, as expected by buildContext.
export function toObsSetNameFunc(obsIndex: string[], labels: string[]) {
  const obsIdToLabel = new Map(obsIndex.map((obsId, i) => [obsId, labels[i]]));
  return (obsId: string) => obsIdToLabel.get(obsId)!;
}

export const motifKey = (motif: string[]) => JSON.stringify(motif.toSorted());

export function expectRelativelyClose(actual: number, expected: number, relTol = 1e-8) {
  if (expected === 0) {
    expect(actual).toBeLessThan(1e-300);
  } else {
    expect(Math.abs(actual - expected) / Math.abs(expected)).toBeLessThan(relTol);
  }
}

// Python reference output types.
export type PyPattern = { support: number, itemsets: string[], neighbor_id?: number[] };
export type PyEnrichment = {
  center: string, motifs: string[], n_center_motif: number, n_center: number, n_motif: number,
  expectation: number, 'p-values': number, 'adj-pval'?: number, if_significant: boolean,
  neighbor_id?: number[], center_id?: number[],
};
export type PyComposition = {
  dataset?: string, center: string, motifs: string[], cell_type: string,
  n_cells: number, n_motif_cells: number, proportion: number,
};

type EnrichmentStats = Pick<MotifEnrichmentResult,
  'center' | 'motifs' | 'nCenterMotif' | 'nCenter' | 'nMotif' | 'expectation' | 'pValue' | 'adjPValue' | 'ifSignificant'>;

// Compare frequent patterns irrespective of the order of patterns with tied support.
// Python cell IDs are integer indices into obsIndex.
export function expectPatternsToMatch(js: FrequentPattern[], py: PyPattern[], obsIndex?: string[]) {
  expect(js.map(p => motifKey(p.itemsets)).toSorted()).toEqual(py.map(p => motifKey(p.itemsets)).toSorted());
  const jsByKey = new Map(js.map(p => [motifKey(p.itemsets), p]));
  for (const pyPattern of py) {
    const jsPattern = jsByKey.get(motifKey(pyPattern.itemsets))!;
    expect(jsPattern.support).toBeCloseTo(pyPattern.support, 12);
    if (pyPattern.neighbor_id && obsIndex) {
      expect(jsPattern.neighborIds!.toSorted()).toEqual(pyPattern.neighbor_id.map(i => obsIndex[i]).toSorted());
    }
  }
  // Sorted by descending support.
  js.slice(1).forEach((p, i) => expect(p.support).toBeLessThanOrEqual(js[i].support));
}

// Compare motif enrichment statistics, irrespective of the order of motifs with tied adjusted p-values.
// Returns the JS results keyed by motif, for further comparisons.
export function expectEnrichmentStatsToMatch<T extends EnrichmentStats>(js: T[], py: PyEnrichment[]): Map<string, T> {
  expect(js.map(r => motifKey(r.motifs)).toSorted()).toEqual(py.map(r => motifKey(r.motifs)).toSorted());
  const jsByKey = new Map(js.map(r => [motifKey(r.motifs), r]));
  for (const pyResult of py) {
    const jsResult = jsByKey.get(motifKey(pyResult.motifs))!;
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
  }
  // Sorted by ascending adjusted p-value when there are multiple motifs.
  if (js.length > 1) {
    js.slice(1).forEach((r, i) => expect(r.adjPValue!).toBeGreaterThanOrEqual(js[i].adjPValue!));
  }
  return jsByKey;
}

// Compare motif composition rows (in order).
export function expectCompositionToMatch(js: (MotifCompositionRow & { datasetId?: string })[], py: PyComposition[]) {
  expect(js.length).toBe(py.length);
  js.forEach((row, i) => {
    const pyRow = py[i];
    expect(row.datasetId).toBe(pyRow.dataset);
    expect(row.center).toBe(pyRow.center);
    expect(row.motifs).toEqual(pyRow.motifs);
    expect(row.cellType).toBe(pyRow.cell_type);
    expect(row.nCells).toBe(pyRow.n_cells);
    expect(row.nMotifCells).toBe(pyRow.n_motif_cells);
    expect(row.proportion).toBeCloseTo(pyRow.proportion, 12);
  });
}
