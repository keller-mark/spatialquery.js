// Port of the multi-FOV class spatial_query_multi.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py
import {
  buildContext,
  findCellIndicesOfType,
  knnTransactions,
  radiusTransactions,
  centerQueryPoints,
  knnNeighborhood,
  radiusNeighborhood,
  countMotifs,
  computeMotifEnrichment,
  normalizeMotifs,
  normalizeMotifList,
  motifCompositionRows,
  assertEnrichmentMode,
  findFpKnn as findFpKnnSingle,
  findFpDist as findFpDistSingle,
  type InitParams,
  type SqContext,
  type MotifsParam,
  type MotifEnrichmentResult,
  type MotifCompositionRow,
  type EnrichmentMode,
  type MotifCounts,
} from './single-fov.js';
import { mineFrequentPatterns, type FrequentPattern } from './fpgrowth.js';
import { nestedMotifDescent } from './nested.js';
import { applyFdrCorrection, fdrCorrection, mannWhitneyU, rankAverage, median } from './stats.js';

export type FovInitParams = InitParams & {
  // Name of the dataset that the FOV belongs to. Multiple FOVs can share a dataset name.
  dataset: string,
};

export type SqMultiContext = {
  // Unique name of each FOV: the dataset name (with '_' replaced by '-') plus a '_<i>' suffix.
  datasets: string[],
  // Single-FOV context of each FOV (each FOV's coordinates are normalized separately).
  contexts: SqContext[],
};

// Build the multi-FOV context object.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L43
export function buildMultiContext(fovs: FovInitParams[]): SqMultiContext {
  const countDict = new Map<string, number>();
  const datasets = fovs.map(({ dataset }) => {
    const name = dataset.replaceAll('_', '-');
    const count = countDict.has(name) ? countDict.get(name)! + 1 : 0;
    countDict.set(name, count);
    return `${name}_${count}`;
  });
  return {
    datasets,
    contexts: fovs.map(fov => buildContext(fov)),
  };
}

// Dataset name of a FOV name ('name_i' -> 'name').
const baseName = (fovName: string) => fovName.split('_')[0];

type Fov = { dataset: string, context: SqContext };

// The FOVs belonging to the selected datasets (all datasets if not provided).
function selectFovs(multi: SqMultiContext, dataset?: string | string[]): Fov[] {
  const validNames = new Set(multi.datasets.map(baseName));
  const selected = dataset === undefined ? validNames : new Set(typeof dataset === 'string' ? [dataset] : dataset);
  for (const ds of selected) {
    if (!validNames.has(ds)) {
      throw new Error(`Invalid input dataset name: ${ds}. Valid dataset names are: ${Array.from(validNames).join(', ')}`);
    }
  }
  return multi.contexts
    .map((context, i) => ({ dataset: multi.datasets[i], context }))
    .filter(fov => selected.has(baseName(fov.dataset)));
}

// Throws if the cell type is absent from every FOV (selected or not, as in Python).
function assertCellTypeInAnyFov(multi: SqMultiContext, ct: string) {
  if (!multi.contexts.some(context => context.labels.includes(ct))) {
    throw new Error(`Found no ${ct} in any datasets!`);
  }
}

// Unique cell type labels across all FOVs.
function allLabels(multi: SqMultiContext): Set<string> {
  return new Set(multi.contexts.flatMap(context => context.labels));
}

// ---------------------------------------------------------------------------
// find_fp_knn / find_fp_dist (pooled across FOVs)
// ---------------------------------------------------------------------------

type DatasetParam = {
  // Datasets to include. Use all datasets if not provided.
  dataset?: string | string[],
};

type MultiFindFpKnnParams = DatasetParam & {
  ct: string,
  k?: number, // Default: 30
  minSupport?: number, // Default: 0.5
  maxDist?: number, // Default: 20
  ifMax?: boolean, // Default: true
};

// Find frequent patterns within the k-NN neighborhoods of a cell type, pooling the
// neighborhoods of all selected FOVs.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L116
export async function multiFindFpKnn(multi: SqMultiContext, params: MultiFindFpKnnParams): Promise<FrequentPattern[]> {
  const { ct, dataset, k = 30, minSupport = 0.5, maxDist = 20, ifMax = true } = params;
  assertCellTypeInAnyFov(multi, ct);
  const transactions = selectFovs(multi, dataset).flatMap(({ context }) => (
    knnTransactions(context, findCellIndicesOfType(context, ct), k, maxDist)
  ));
  return mineFrequentPatterns(transactions, minSupport, ifMax);
}

type MultiFindFpDistParams = DatasetParam & {
  ct: string,
  maxDist?: number, // Default: 20
  minSize?: number, // Default: 0
  minSupport?: number, // Default: 0.5
  ifMax?: boolean, // Default: true
};

// Find frequent patterns within the radius-based neighborhoods of a cell type, pooling the
// neighborhoods of all selected FOVs.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L219
export async function multiFindFpDist(multi: SqMultiContext, params: MultiFindFpDistParams): Promise<FrequentPattern[]> {
  const { ct, dataset, maxDist = 20, minSize = 0, minSupport = 0.5, ifMax = true } = params;
  assertCellTypeInAnyFov(multi, ct);
  const transactions = selectFovs(multi, dataset).flatMap(({ context }) => {
    const points = centerQueryPoints(context, findCellIndicesOfType(context, ct));
    return radiusTransactions(context, points, maxDist, minSize).transactions;
  });
  return mineFrequentPatterns(transactions, minSupport, ifMax);
}

type FindFpFovParams = {
  ct: string,
  // FOV name, in the 'name_i' format of SqMultiContext.datasets.
  datasetI: string,
};

function getFov(multi: SqMultiContext, datasetI: string): SqContext {
  const index = multi.datasets.indexOf(datasetI);
  if (index === -1) {
    throw new Error(`Found no ${baseName(datasetI)} in any datasets.`);
  }
  return multi.contexts[index];
}

// Find frequent (maximal) patterns within the k-NN neighborhoods of a cell type in a single FOV.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L993
export async function findFpKnnFov(
  multi: SqMultiContext,
  params: FindFpFovParams & { k?: number, minSupport?: number, maxDist?: number },
): Promise<FrequentPattern[]> {
  const { ct, datasetI, k = 30, minSupport = 0.5, maxDist = 20 } = params;
  const context = getFov(multi, datasetI);
  if (!context.labels.includes(ct)) return [];
  return findFpKnnSingle(context, { ct, k, minSupport, maxDist });
}

// Find frequent (maximal) patterns within the radius-based neighborhoods of a cell type in a single FOV.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L1049
export async function findFpDistFov(
  multi: SqMultiContext,
  params: FindFpFovParams & { maxDist?: number, minSize?: number, minSupport?: number },
): Promise<FrequentPattern[]> {
  const { ct, datasetI, maxDist = 20, minSize = 0, minSupport = 0.5 } = params;
  const context = getFov(multi, datasetI);
  if (!context.labels.includes(ct)) return [];
  return findFpDistSingle(context, { ct, maxDist, minSize, minSupport });
}

// ---------------------------------------------------------------------------
// Motif enrichment (counts pooled across FOVs)
// ---------------------------------------------------------------------------

export type MultiFovMotifEnrichmentResult = Omit<MotifEnrichmentResult, 'neighborIds' | 'centerIds'> & {
  // Obs IDs of neighbor cells with motif cell types, per FOV name (only if returnCellIds is true).
  // Only FOVs with at least one center cell with the motif are included.
  neighborIds?: Record<string, string[]>,
  // Obs IDs of center cells with the motif in their neighborhood, per FOV name (only if returnCellIds is true).
  centerIds?: Record<string, string[]>,
};

// Normalize user-specified motifs against the labels of all FOVs, throwing if none remain.
function normalizeMultiMotifs(multi: SqMultiContext, motifs: MotifsParam): string[][] {
  const filtered = normalizeMotifs(motifs, allLabels(multi));
  if (filtered.length === 0) {
    throw new Error("All cell types in motifs are missing in the cell type labels.");
  }
  return filtered;
}

function multiMotifEnrichment(
  fovs: Fov[],
  ct: string,
  motifs: string[][],
  neighborhood: (context: SqContext) => (i: number) => number[],
  returnCellIds: boolean,
): MultiFovMotifEnrichmentResult[] {
  const totals = motifs.map(() => ({ numCells: 0, numCenter: 0, numMotif: 0, numCenterMotif: 0 }));
  const neighborIds = motifs.map(() => ({} as Record<string, string[]>));
  const centerIds = motifs.map(() => ({} as Record<string, string[]>));

  for (const { dataset, context } of fovs) {
    const counts: MotifCounts[] = countMotifs(context, ct, motifs, neighborhood(context), returnCellIds);
    counts.forEach((c, m) => {
      totals[m].numCells += c.numCells;
      totals[m].numCenter += c.numCenter;
      totals[m].numMotif += c.numMotif;
      totals[m].numCenterMotif += c.numCenterMotif;
      if (returnCellIds && c.centerIndices!.length > 0) {
        neighborIds[m][dataset] = c.neighborIndices!.map(j => context.obsIndex[j]);
        centerIds[m][dataset] = c.centerIndices!.map(i => context.obsIndex[i]);
      }
    });
  }

  const out = motifs.map((motif, m): MultiFovMotifEnrichmentResult => {
    const { neighborIds: _n, centerIds: _c, ...result } = computeMotifEnrichment({ center: ct, motif, ...totals[m] });
    return returnCellIds
      ? { ...result, neighborIds: neighborIds[m], centerIds: centerIds[m] }
      : result;
  });
  return applyFdrCorrection(out);
}

type MultiMotifEnrichmentKnnParams = DatasetParam & {
  ct: string,
  motifs?: MotifsParam,
  k?: number, // Default: 30
  minSupport?: number, // Default: 0.5
  maxDist?: number, // Default: 20
  returnCellIds?: boolean, // Default: false
  mode?: EnrichmentMode, // Default: 'maximal'
};

// Perform motif enrichment analysis using k-nearest neighbors in multiple FOVs.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L319
export async function multiMotifEnrichmentKnn(
  multi: SqMultiContext,
  params: MultiMotifEnrichmentKnnParams,
): Promise<MultiFovMotifEnrichmentResult[]> {
  const { ct, dataset, k = 30, minSupport = 0.5, maxDist = 20, returnCellIds = false, mode = 'maximal' } = params;
  const fovs = selectFovs(multi, dataset);
  assertCellTypeInAnyFov(multi, ct);
  assertEnrichmentMode(mode);

  if (mode === 'nested' && params.motifs === undefined) {
    return nestedMotifDescent({
      testMotifs: motifs => multiMotifEnrichmentKnn(multi, { ...params, motifs, mode: 'maximal' }),
      frequentPatterns: await multiFindFpKnn(multi, { ct, dataset, k, minSupport, maxDist, ifMax: false }),
    });
  }

  const motifs = params.motifs === undefined
    ? (await multiFindFpKnn(multi, { ct, dataset, k, minSupport, maxDist })).map(p => p.itemsets)
    : normalizeMultiMotifs(multi, params.motifs);
  return multiMotifEnrichment(fovs, ct, motifs, context => knnNeighborhood(context, k, maxDist), returnCellIds);
}

type MultiMotifEnrichmentDistParams = DatasetParam & {
  ct: string,
  motifs?: MotifsParam,
  maxDist?: number, // Default: 20
  minSize?: number, // Default: 0
  minSupport?: number, // Default: 0.5
  returnCellIds?: boolean, // Default: false
  mode?: EnrichmentMode, // Default: 'maximal'
};

// Perform motif enrichment analysis within a radius-based neighborhood in multiple FOVs.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L581
export async function multiMotifEnrichmentDist(
  multi: SqMultiContext,
  params: MultiMotifEnrichmentDistParams,
): Promise<MultiFovMotifEnrichmentResult[]> {
  const { ct, dataset, maxDist = 20, minSize = 0, minSupport = 0.5, returnCellIds = false, mode = 'maximal' } = params;
  const fovs = selectFovs(multi, dataset);
  assertCellTypeInAnyFov(multi, ct);
  assertEnrichmentMode(mode);

  if (mode === 'nested' && params.motifs === undefined) {
    return nestedMotifDescent({
      testMotifs: motifs => multiMotifEnrichmentDist(multi, { ...params, motifs, mode: 'maximal' }),
      frequentPatterns: await multiFindFpDist(multi, { ct, dataset, maxDist, minSize, minSupport, ifMax: false }),
    });
  }

  const motifs = params.motifs === undefined
    ? (await multiFindFpDist(multi, { ct, dataset, maxDist, minSize, minSupport })).map(p => p.itemsets)
    : normalizeMultiMotifs(multi, params.motifs);
  return multiMotifEnrichment(fovs, ct, motifs, context => radiusNeighborhood(context, maxDist), returnCellIds);
}

// ---------------------------------------------------------------------------
// Motif composition
// ---------------------------------------------------------------------------

export type MultiFovMotifCompositionRow = MotifCompositionRow & {
  // FOV name that the center cell belongs to.
  dataset: string,
};

// Pool the motif composition over the selected FOVs.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L851
function multiMotifComposition(
  multi: SqMultiContext,
  ct: string,
  motifsParam: MotifsParam,
  dataset: string | string[] | undefined,
  neighborhood: (context: SqContext) => (i: number) => number[],
): MultiFovMotifCompositionRow[] {
  const fovs = selectFovs(multi, dataset);
  assertCellTypeInAnyFov(multi, ct);
  const motifs = normalizeMotifList(motifsParam, allLabels(multi));

  return fovs.flatMap(({ dataset: fovName, context }) => {
    const cinds = findCellIndicesOfType(context, ct);
    const labelsUnique = new Set(context.labels);
    // A motif with a cell type missing from this FOV has no motif-positive center here.
    const active = motifs.filter(m => m.every(t => labelsUnique.has(t)));
    if (cinds.length === 0 || active.length === 0) return [];
    return motifCompositionRows(context, ct, active, cinds, neighborhood(context))
      .map(row => ({ dataset: fovName, ...row }));
  });
}

// Compute the composition of motif cell types in the k-NN neighborhoods of center cells across multiple FOVs.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L931
export function multiMotifCompositionKnn(
  multi: SqMultiContext,
  params: DatasetParam & { ct: string, motifs: MotifsParam, k?: number, maxDist?: number },
): MultiFovMotifCompositionRow[] {
  const { ct, motifs, dataset, k = 30, maxDist = 20 } = params;
  return multiMotifComposition(multi, ct, motifs, dataset, context => knnNeighborhood(context, k, maxDist));
}

// Compute the composition of motif cell types in the radius-based neighborhoods of center cells across multiple FOVs.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L888
export function multiMotifCompositionDist(
  multi: SqMultiContext,
  params: DatasetParam & { ct: string, motifs: MotifsParam, maxDist?: number },
): MultiFovMotifCompositionRow[] {
  const { ct, motifs, dataset, maxDist = 20 } = params;
  return multiMotifComposition(multi, ct, motifs, dataset, context => radiusNeighborhood(context, maxDist));
}

// ---------------------------------------------------------------------------
// Differential analysis of motif patterns between two datasets
// ---------------------------------------------------------------------------

export type DifferentialTestResult = {
  // Sorted motif cell types.
  itemsets: string[],
  // Support of the motif in each FOV of each of the two datasets.
  supports: Record<string, number[]>,
  // Mean support across the FOVs of each of the two datasets.
  supportMean: Record<string, number>,
  // Two-sided Mann-Whitney U test p-value.
  pValue: number,
  // FDR-corrected p-value (equal to pValue when a single motif is tested).
  adjPValue: number,
  ifSignificant: boolean,
  // Dataset with the higher median rank of support ('neither' if a dataset has no FOVs).
  datasetHigherFrequency: string,
};

export type DifferentialResult = Pick<DifferentialTestResult, 'itemsets' | 'supportMean' | 'adjPValue'>;

// Normalize motifs as differential_analysis_motif_* does (no filtering of unknown cell types).
function toDifferentialMotifs(motifs: MotifsParam): string[][] {
  if (typeof motifs === 'string') return [[motifs]];
  if (motifs.length > 0 && typeof motifs[0] === 'string') return [motifs as string[]];
  return motifs as string[][];
}

// Test every motif for a difference in support between the FOVs of two datasets.
// Returns all tested motifs, unlike differentialAnalysisKnn/Dist which return only the significant ones.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_differential_pattern.py
function differentialTests(
  multi: SqMultiContext,
  datasets: string[],
  ct: string,
  motifsParam: MotifsParam,
  // Support of each motif among the given center cells of a FOV.
  supportsInFov: (context: SqContext, cinds: number[], motifs: string[][]) => number[],
): DifferentialTestResult[] {
  if (datasets.length !== 2) {
    throw new Error("Require 2 datasets for differential analysis.");
  }
  const motifs: string[][] = [];
  const motifKeys = new Set<string>();
  for (const motif of toDifferentialMotifs(motifsParam)) {
    const sorted = motif.toSorted();
    const key = JSON.stringify(sorted);
    if (!motifKeys.has(key)) {
      motifKeys.add(key);
      motifs.push(sorted);
    }
  }

  const supports = motifs.map(() => Object.fromEntries(datasets.map(d => [d, [] as number[]])));
  for (const datasetName of datasets) {
    for (const { context } of selectFovs(multi, datasetName)) {
      const cinds = findCellIndicesOfType(context, ct);
      const fovSupports = cinds.length === 0 ? motifs.map(() => 0) : supportsInFov(context, cinds, motifs);
      fovSupports.forEach((s, m) => supports[m][datasetName].push(s));
    }
  }

  const [d0, d1] = datasets;
  const results = motifs.map((itemsets, m) => {
    const group1 = supports[m][d0];
    const group2 = supports[m][d1];
    const mean = (g: number[]) => g.reduce((a, b) => a + b, 0) / g.length;
    let pValue = 1.0;
    let datasetHigherFrequency = 'neither';
    if (group1.length > 0 && group2.length > 0) {
      pValue = mannWhitneyU(group1, group2).pValue;
      // Determine which dataset has higher frequency using the median rank.
      const { ranks } = rankAverage([...group1, ...group2]);
      const medianRank1 = median(ranks.slice(0, group1.length));
      const medianRank2 = median(ranks.slice(group1.length));
      datasetHigherFrequency = medianRank1 > medianRank2 ? d0 : d1;
    }
    return {
      itemsets,
      supports: supports[m],
      supportMean: { [d0]: mean(group1), [d1]: mean(group2) },
      pValue,
      datasetHigherFrequency,
    };
  });

  if (results.length > 1) {
    const { rejected, adjusted } = fdrCorrection(results.map(r => r.pValue), 0.05);
    return results.map((r, i) => ({ ...r, adjPValue: adjusted[i], ifSignificant: rejected[i] }));
  }
  return results.map(r => ({ ...r, adjPValue: r.pValue, ifSignificant: r.pValue < 0.05 }));
}

// Split the significant results by the dataset with the higher frequency, sorted by adjusted p-value.
function splitSignificant(results: DifferentialTestResult[], datasets: string[]): Record<string, DifferentialResult[]> {
  return Object.fromEntries(datasets.map(d => [
    d,
    results
      .filter(r => r.datasetHigherFrequency === d && r.ifSignificant)
      .map(({ itemsets, supportMean, adjPValue }) => ({ itemsets, supportMean, adjPValue }))
      .toSorted((a, b) => a.adjPValue - b.adjPValue),
  ]));
}

// Union of the frequent patterns discovered in each FOV of the two datasets.
async function discoverMotifs(
  multi: SqMultiContext,
  datasets: string[],
  findFpFov: (datasetI: string) => Promise<FrequentPattern[]>,
): Promise<string[][]> {
  const motifs = new Map<string, string[]>();
  for (const d of datasets) {
    for (const datasetI of multi.datasets.filter(ds => baseName(ds) === d)) {
      for (const { itemsets } of await findFpFov(datasetI)) {
        motifs.set(JSON.stringify(itemsets.toSorted()), itemsets.toSorted());
      }
    }
  }
  return Array.from(motifs.values());
}

function validateDifferentialDatasets(multi: SqMultiContext, datasets: string[]) {
  if (datasets.length !== 2) {
    throw new Error("Require 2 datasets for differential analysis.");
  }
  selectFovs(multi, datasets);
}

// Fraction of the center cells whose neighborhood contains every motif cell type
// (0 for motifs with a cell type missing from the FOV).
function motifSupports(
  context: SqContext,
  cinds: number[],
  motifs: string[][],
  neighborsOf: (i: number) => number[],
  isValidCenter: (neighbors: number[]) => boolean = () => true,
): number[] {
  const labelsUnique = new Set(context.labels);
  const neighborTypes = cinds.map((i) => {
    const neighbors = neighborsOf(i);
    return isValidCenter(neighbors) ? new Set(neighbors.map(j => context.labels[j])) : null;
  });
  return motifs.map((motif) => {
    if (!motif.every(t => labelsUnique.has(t))) return 0;
    const count = neighborTypes.filter(types => types !== null && motif.every(t => types.has(t))).length;
    return count / cinds.length;
  });
}

type DifferentialAnalysisKnnParams = {
  ct: string,
  // Exactly 2 dataset names to compare.
  datasets: string[],
  // Motifs to test. If not provided, the frequent patterns found in each FOV of both datasets are tested.
  motifs?: MotifsParam,
  k?: number, // Default: 30
  minSupport?: number, // Default: 0.5 (only used when motifs is not provided)
  maxDist?: number, // Default: 20
};

// Test all motifs for differential support between two datasets using k-NN neighborhoods.
// Returns every tested motif (including non-significant ones).
export async function differentialTestsKnn(multi: SqMultiContext, params: DifferentialAnalysisKnnParams): Promise<DifferentialTestResult[]> {
  const { ct, datasets, k = 30, minSupport = 0.5, maxDist = 20 } = params;
  validateDifferentialDatasets(multi, datasets);
  const motifs = params.motifs ?? await discoverMotifs(multi, datasets, datasetI => (
    findFpKnnFov(multi, { ct, datasetI, k, minSupport, maxDist })
  ));
  return differentialTests(multi, datasets, ct, motifs, (context, cinds, ms) => (
    motifSupports(context, cinds, ms, knnNeighborhood(context, k, maxDist))
  ));
}

// Perform differential analysis of motif patterns between two datasets using k-NN neighborhoods.
// Returns, for each dataset, the motifs significantly more frequent in that dataset.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L1105
export async function differentialAnalysisKnn(
  multi: SqMultiContext,
  params: DifferentialAnalysisKnnParams,
): Promise<Record<string, DifferentialResult[]>> {
  return splitSignificant(await differentialTestsKnn(multi, params), params.datasets);
}

type DifferentialAnalysisDistParams = {
  ct: string,
  // Exactly 2 dataset names to compare.
  datasets: string[],
  // Motifs to test. If not provided, the frequent patterns found in each FOV of both datasets are tested.
  motifs?: MotifsParam,
  maxDist?: number, // Default: 20
  minSupport?: number, // Default: 0.5 (only used when motifs is not provided)
  // Minimum neighborhood size for each center cell to be considered.
  minSize?: number, // Default: 0
};

// Test all motifs for differential support between two datasets using radius-based neighborhoods.
// Returns every tested motif (including non-significant ones).
export async function differentialTestsDist(multi: SqMultiContext, params: DifferentialAnalysisDistParams): Promise<DifferentialTestResult[]> {
  const { ct, datasets, maxDist = 20, minSupport = 0.5, minSize = 0 } = params;
  validateDifferentialDatasets(multi, datasets);
  const motifs = params.motifs ?? await discoverMotifs(multi, datasets, datasetI => (
    findFpDistFov(multi, { ct, datasetI, maxDist, minSize, minSupport })
  ));
  // Center cells with neighborhoods not larger than minSize never count as having the motif,
  // but still count in the denominator of the support.
  return differentialTests(multi, datasets, ct, motifs, (context, cinds, ms) => (
    motifSupports(context, cinds, ms, radiusNeighborhood(context, maxDist), neighbors => neighbors.length > minSize)
  ));
}

// Perform differential analysis of motif patterns between two datasets using radius-based neighborhoods.
// Returns, for each dataset, the motifs significantly more frequent in that dataset.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L1206
export async function differentialAnalysisDist(
  multi: SqMultiContext,
  params: DifferentialAnalysisDistParams,
): Promise<Record<string, DifferentialResult[]>> {
  return splitSignificant(await differentialTestsDist(multi, params), params.datasets);
}
