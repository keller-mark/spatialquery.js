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
import { applyFdrCorrection, fdrCorrection, mannWhitneyU, rankAverage, median, mean } from './stats.js';

export type FovInitParams = InitParams & {
  // Unique ID of the dataset (FOV).
  datasetId: string,
  // Class (e.g., condition) that the dataset belongs to. Multiple datasets can share a class.
  datasetClass: string,
};

export type SqMultiContext = {
  // Unique ID of each dataset (FOV).
  datasetIds: string[],
  // Class of each dataset (FOV).
  datasetClasses: string[],
  // Single-FOV context of each dataset (each FOV's coordinates are normalized separately).
  contexts: SqContext[],
};

// Build the multi-FOV context object.
// Unlike Python, which derives unique FOV names from the (possibly repeated) dataset names
// ('name' -> 'name_0', 'name_1', ...), each FOV has an explicit ID and class.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L43
export function buildMultiContext(fovs: FovInitParams[]): SqMultiContext {
  const datasetIds = fovs.map(fov => fov.datasetId);
  const duplicates = datasetIds.filter((id, i) => datasetIds.indexOf(id) !== i);
  if (duplicates.length > 0) {
    throw new Error(`Dataset IDs must be unique. Found duplicates: ${Array.from(new Set(duplicates)).join(', ')}`);
  }
  return {
    datasetIds,
    datasetClasses: fovs.map(fov => fov.datasetClass),
    contexts: fovs.map(fov => buildContext(fov)),
  };
}

type Fov = { datasetId: string, context: SqContext };

// The datasets (FOVs) belonging to the selected classes (all classes if not provided).
function selectFovs(multi: SqMultiContext, datasetClass?: string | string[]): Fov[] {
  const validClasses = new Set(multi.datasetClasses);
  const selected = datasetClass === undefined
    ? validClasses
    : new Set(typeof datasetClass === 'string' ? [datasetClass] : datasetClass);
  for (const c of selected) {
    if (!validClasses.has(c)) {
      throw new Error(`Invalid dataset class: ${c}. Valid dataset classes are: ${Array.from(validClasses).join(', ')}`);
    }
  }
  return multi.contexts
    .map((context, i) => ({ datasetId: multi.datasetIds[i], datasetClass: multi.datasetClasses[i], context }))
    .filter(fov => selected.has(fov.datasetClass));
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

type DatasetClassParam = {
  // Classes of the datasets (FOVs) to include. Use all datasets if not provided.
  datasetClass?: string | string[],
};

type MultiFindFpKnnParams = DatasetClassParam & {
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
  const { ct, datasetClass, k = 30, minSupport = 0.5, maxDist = 20, ifMax = true } = params;
  assertCellTypeInAnyFov(multi, ct);
  const transactions = selectFovs(multi, datasetClass).flatMap(({ context }) => (
    knnTransactions(context, findCellIndicesOfType(context, ct), k, maxDist)
  ));
  return mineFrequentPatterns(transactions, minSupport, ifMax);
}

type MultiFindFpDistParams = DatasetClassParam & {
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
  const { ct, datasetClass, maxDist = 20, minSize = 0, minSupport = 0.5, ifMax = true } = params;
  assertCellTypeInAnyFov(multi, ct);
  const transactions = selectFovs(multi, datasetClass).flatMap(({ context }) => {
    const points = centerQueryPoints(context, findCellIndicesOfType(context, ct));
    return radiusTransactions(context, points, maxDist, minSize).transactions;
  });
  return mineFrequentPatterns(transactions, minSupport, ifMax);
}

type FindFpFovParams = {
  ct: string,
  // ID of the dataset (FOV).
  datasetId: string,
};

function getFov(multi: SqMultiContext, datasetId: string): SqContext {
  const index = multi.datasetIds.indexOf(datasetId);
  if (index === -1) {
    throw new Error(`Found no dataset with ID ${datasetId}.`);
  }
  return multi.contexts[index];
}

// Find frequent (maximal) patterns within the k-NN neighborhoods of a cell type in a single FOV.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L993
export async function findFpKnnFov(
  multi: SqMultiContext,
  params: FindFpFovParams & { k?: number, minSupport?: number, maxDist?: number },
): Promise<FrequentPattern[]> {
  const { ct, datasetId, k = 30, minSupport = 0.5, maxDist = 20 } = params;
  const context = getFov(multi, datasetId);
  if (!context.labels.includes(ct)) return [];
  return findFpKnnSingle(context, { ct, k, minSupport, maxDist });
}

// Find frequent (maximal) patterns within the radius-based neighborhoods of a cell type in a single FOV.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L1049
export async function findFpDistFov(
  multi: SqMultiContext,
  params: FindFpFovParams & { maxDist?: number, minSize?: number, minSupport?: number },
): Promise<FrequentPattern[]> {
  const { ct, datasetId, maxDist = 20, minSize = 0, minSupport = 0.5 } = params;
  const context = getFov(multi, datasetId);
  if (!context.labels.includes(ct)) return [];
  return findFpDistSingle(context, { ct, maxDist, minSize, minSupport });
}

// ---------------------------------------------------------------------------
// Motif enrichment (counts pooled across FOVs)
// ---------------------------------------------------------------------------

export type MultiFovMotifEnrichmentResult = Omit<MotifEnrichmentResult, 'neighborIds' | 'centerIds'> & {
  // Obs IDs of neighbor cells with motif cell types, per dataset ID (only if returnCellIds is true).
  // Only datasets with at least one center cell with the motif are included.
  neighborIds?: Record<string, string[]>,
  // Obs IDs of center cells with the motif in their neighborhood, per dataset ID (only if returnCellIds is true).
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

  for (const { datasetId, context } of fovs) {
    const counts: MotifCounts[] = countMotifs(context, ct, motifs, neighborhood(context), returnCellIds);
    counts.forEach((c, m) => {
      totals[m].numCells += c.numCells;
      totals[m].numCenter += c.numCenter;
      totals[m].numMotif += c.numMotif;
      totals[m].numCenterMotif += c.numCenterMotif;
      if (returnCellIds && c.centerIndices!.length > 0) {
        neighborIds[m][datasetId] = c.neighborIndices!.map(j => context.obsIndex[j]);
        centerIds[m][datasetId] = c.centerIndices!.map(i => context.obsIndex[i]);
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

type MultiMotifEnrichmentKnnParams = DatasetClassParam & {
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
  const { ct, datasetClass, k = 30, minSupport = 0.5, maxDist = 20, returnCellIds = false, mode = 'maximal' } = params;
  const fovs = selectFovs(multi, datasetClass);
  assertCellTypeInAnyFov(multi, ct);
  assertEnrichmentMode(mode);

  if (mode === 'nested' && params.motifs === undefined) {
    return nestedMotifDescent({
      testMotifs: motifs => multiMotifEnrichmentKnn(multi, { ...params, motifs, mode: 'maximal' }),
      frequentPatterns: await multiFindFpKnn(multi, { ct, datasetClass, k, minSupport, maxDist, ifMax: false }),
    });
  }

  const motifs = params.motifs === undefined
    ? (await multiFindFpKnn(multi, { ct, datasetClass, k, minSupport, maxDist })).map(p => p.itemsets)
    : normalizeMultiMotifs(multi, params.motifs);
  return multiMotifEnrichment(fovs, ct, motifs, context => knnNeighborhood(context, k, maxDist), returnCellIds);
}

type MultiMotifEnrichmentDistParams = DatasetClassParam & {
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
  const { ct, datasetClass, maxDist = 20, minSize = 0, minSupport = 0.5, returnCellIds = false, mode = 'maximal' } = params;
  const fovs = selectFovs(multi, datasetClass);
  assertCellTypeInAnyFov(multi, ct);
  assertEnrichmentMode(mode);

  if (mode === 'nested' && params.motifs === undefined) {
    return nestedMotifDescent({
      testMotifs: motifs => multiMotifEnrichmentDist(multi, { ...params, motifs, mode: 'maximal' }),
      frequentPatterns: await multiFindFpDist(multi, { ct, datasetClass, maxDist, minSize, minSupport, ifMax: false }),
    });
  }

  const motifs = params.motifs === undefined
    ? (await multiFindFpDist(multi, { ct, datasetClass, maxDist, minSize, minSupport })).map(p => p.itemsets)
    : normalizeMultiMotifs(multi, params.motifs);
  return multiMotifEnrichment(fovs, ct, motifs, context => radiusNeighborhood(context, maxDist), returnCellIds);
}

// ---------------------------------------------------------------------------
// Motif composition
// ---------------------------------------------------------------------------

export type MultiFovMotifCompositionRow = MotifCompositionRow & {
  // ID of the dataset (FOV) that the center cell belongs to.
  datasetId: string,
};

// Pool the motif composition over the selected FOVs.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L851
function multiMotifComposition(
  multi: SqMultiContext,
  ct: string,
  motifsParam: MotifsParam,
  datasetClass: string | string[] | undefined,
  neighborhood: (context: SqContext) => (i: number) => number[],
): MultiFovMotifCompositionRow[] {
  const fovs = selectFovs(multi, datasetClass);
  assertCellTypeInAnyFov(multi, ct);
  const motifs = normalizeMotifList(motifsParam, allLabels(multi));

  return fovs.flatMap(({ datasetId, context }) => {
    const cinds = findCellIndicesOfType(context, ct);
    const labelsUnique = new Set(context.labels);
    // A motif with a cell type missing from this FOV has no motif-positive center here.
    const active = motifs.filter(m => m.every(t => labelsUnique.has(t)));
    if (cinds.length === 0 || active.length === 0) return [];
    return motifCompositionRows(context, ct, active, cinds, neighborhood(context))
      .map(row => ({ datasetId, ...row }));
  });
}

// Compute the composition of motif cell types in the k-NN neighborhoods of center cells across multiple FOVs.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L931
export function multiMotifCompositionKnn(
  multi: SqMultiContext,
  params: DatasetClassParam & { ct: string, motifs: MotifsParam, k?: number, maxDist?: number },
): MultiFovMotifCompositionRow[] {
  const { ct, motifs, datasetClass, k = 30, maxDist = 20 } = params;
  return multiMotifComposition(multi, ct, motifs, datasetClass, context => knnNeighborhood(context, k, maxDist));
}

// Compute the composition of motif cell types in the radius-based neighborhoods of center cells across multiple FOVs.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L888
export function multiMotifCompositionDist(
  multi: SqMultiContext,
  params: DatasetClassParam & { ct: string, motifs: MotifsParam, maxDist?: number },
): MultiFovMotifCompositionRow[] {
  const { ct, motifs, datasetClass, maxDist = 20 } = params;
  return multiMotifComposition(multi, ct, motifs, datasetClass, context => radiusNeighborhood(context, maxDist));
}

// ---------------------------------------------------------------------------
// Differential analysis of motif patterns between two dataset classes
// ---------------------------------------------------------------------------

export type DifferentialTestResult = {
  // Sorted motif cell types.
  itemsets: string[],
  // Support of the motif in each dataset (FOV) of each of the two classes.
  supports: Record<string, number[]>,
  // Mean support across the datasets of each of the two classes.
  supportMean: Record<string, number>,
  // Two-sided Mann-Whitney U test p-value.
  pValue: number,
  // FDR-corrected p-value (equal to pValue when a single motif is tested).
  adjPValue: number,
  ifSignificant: boolean,
  // Class with the higher median rank of support ('neither' if a class has no datasets).
  datasetClassHigherFrequency: string,
};

export type DifferentialResult = Pick<DifferentialTestResult, 'itemsets' | 'supportMean' | 'adjPValue'>;

// Normalize motifs as differential_analysis_motif_* does (no filtering of unknown cell types).
function toDifferentialMotifs(motifs: MotifsParam): string[][] {
  if (typeof motifs === 'string') return [[motifs]];
  if (motifs.length > 0 && typeof motifs[0] === 'string') return [motifs as string[]];
  return motifs as string[][];
}

// Test every motif for a difference in support between the datasets of two classes.
// Returns all tested motifs, unlike differentialAnalysisKnn/Dist which return only the significant ones.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_differential_pattern.py
function differentialTests(
  multi: SqMultiContext,
  datasetClasses: string[],
  ct: string,
  motifsParam: MotifsParam,
  // Support of each motif among the given center cells of a dataset.
  supportsInFov: (context: SqContext, cinds: number[], motifs: string[][]) => number[],
): DifferentialTestResult[] {
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

  const supports = motifs.map(() => Object.fromEntries(datasetClasses.map(c => [c, [] as number[]])));
  for (const datasetClass of datasetClasses) {
    for (const { context } of selectFovs(multi, datasetClass)) {
      const cinds = findCellIndicesOfType(context, ct);
      const fovSupports = cinds.length === 0 ? motifs.map(() => 0) : supportsInFov(context, cinds, motifs);
      fovSupports.forEach((s, m) => supports[m][datasetClass].push(s));
    }
  }

  const [c0, c1] = datasetClasses;
  const results = motifs.map((itemsets, m) => {
    const group1 = supports[m][c0];
    const group2 = supports[m][c1];
    let pValue = 1.0;
    let datasetClassHigherFrequency = 'neither';
    if (group1.length > 0 && group2.length > 0) {
      pValue = mannWhitneyU(group1, group2).pValue;
      // Determine which class has higher frequency using the median rank.
      const ranks = rankAverage([...group1, ...group2]);
      const medianRank1 = median(ranks.slice(0, group1.length));
      const medianRank2 = median(ranks.slice(group1.length));
      datasetClassHigherFrequency = medianRank1 > medianRank2 ? c0 : c1;
    }
    return {
      itemsets,
      supports: supports[m],
      supportMean: { [c0]: mean(group1), [c1]: mean(group2) },
      pValue,
      datasetClassHigherFrequency,
    };
  });

  if (results.length > 1) {
    const { rejected, adjusted } = fdrCorrection(results.map(r => r.pValue), 0.05);
    return results.map((r, i) => ({ ...r, adjPValue: adjusted[i], ifSignificant: rejected[i] }));
  }
  return results.map(r => ({ ...r, adjPValue: r.pValue, ifSignificant: r.pValue < 0.05 }));
}

// Split the significant results by the class with the higher frequency, sorted by adjusted p-value.
function splitSignificant(results: DifferentialTestResult[], datasetClasses: string[]): Record<string, DifferentialResult[]> {
  return Object.fromEntries(datasetClasses.map(c => [
    c,
    results
      .filter(r => r.datasetClassHigherFrequency === c && r.ifSignificant)
      .map(({ itemsets, supportMean, adjPValue }) => ({ itemsets, supportMean, adjPValue }))
      .toSorted((a, b) => a.adjPValue - b.adjPValue),
  ]));
}

// Union of the frequent patterns discovered in each dataset of the two classes.
async function discoverMotifs(
  multi: SqMultiContext,
  datasetClasses: string[],
  findFpFov: (datasetId: string) => Promise<FrequentPattern[]>,
): Promise<string[][]> {
  const motifs = new Map<string, string[]>();
  for (const datasetClass of datasetClasses) {
    for (const { datasetId } of selectFovs(multi, datasetClass)) {
      for (const { itemsets } of await findFpFov(datasetId)) {
        motifs.set(JSON.stringify(itemsets.toSorted()), itemsets.toSorted());
      }
    }
  }
  return Array.from(motifs.values());
}

function validateDifferentialClasses(multi: SqMultiContext, datasetClasses: string[]) {
  if (datasetClasses.length !== 2) {
    throw new Error("Require 2 dataset classes for differential analysis.");
  }
  selectFovs(multi, datasetClasses);
}

// Fraction of the center cells whose neighborhood contains every motif cell type
// (0 for motifs with a cell type missing from the dataset).
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
  // Exactly 2 dataset classes to compare.
  datasetClasses: string[],
  // Motifs to test. If not provided, the frequent patterns found in each dataset of both classes are tested.
  motifs?: MotifsParam,
  k?: number, // Default: 30
  minSupport?: number, // Default: 0.5 (only used when motifs is not provided)
  maxDist?: number, // Default: 20
};

// Test all motifs for differential support between two dataset classes using k-NN neighborhoods.
// Returns every tested motif (including non-significant ones).
export async function differentialTestsKnn(multi: SqMultiContext, params: DifferentialAnalysisKnnParams): Promise<DifferentialTestResult[]> {
  const { ct, datasetClasses, k = 30, minSupport = 0.5, maxDist = 20 } = params;
  validateDifferentialClasses(multi, datasetClasses);
  const motifs = params.motifs ?? await discoverMotifs(multi, datasetClasses, datasetId => (
    findFpKnnFov(multi, { ct, datasetId, k, minSupport, maxDist })
  ));
  return differentialTests(multi, datasetClasses, ct, motifs, (context, cinds, ms) => (
    motifSupports(context, cinds, ms, knnNeighborhood(context, k, maxDist))
  ));
}

// Perform differential analysis of motif patterns between two dataset classes using k-NN neighborhoods.
// Returns, for each class, the motifs significantly more frequent in that class.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L1105
export async function differentialAnalysisKnn(
  multi: SqMultiContext,
  params: DifferentialAnalysisKnnParams,
): Promise<Record<string, DifferentialResult[]>> {
  return splitSignificant(await differentialTestsKnn(multi, params), params.datasetClasses);
}

type DifferentialAnalysisDistParams = {
  ct: string,
  // Exactly 2 dataset classes to compare.
  datasetClasses: string[],
  // Motifs to test. If not provided, the frequent patterns found in each dataset of both classes are tested.
  motifs?: MotifsParam,
  maxDist?: number, // Default: 20
  minSupport?: number, // Default: 0.5 (only used when motifs is not provided)
  // Minimum neighborhood size for each center cell to be considered.
  minSize?: number, // Default: 0
};

// Test all motifs for differential support between two dataset classes using radius-based neighborhoods.
// Returns every tested motif (including non-significant ones).
export async function differentialTestsDist(multi: SqMultiContext, params: DifferentialAnalysisDistParams): Promise<DifferentialTestResult[]> {
  const { ct, datasetClasses, maxDist = 20, minSupport = 0.5, minSize = 0 } = params;
  validateDifferentialClasses(multi, datasetClasses);
  const motifs = params.motifs ?? await discoverMotifs(multi, datasetClasses, datasetId => (
    findFpDistFov(multi, { ct, datasetId, maxDist, minSize, minSupport })
  ));
  // Center cells with neighborhoods not larger than minSize never count as having the motif,
  // but still count in the denominator of the support.
  return differentialTests(multi, datasetClasses, ct, motifs, (context, cinds, ms) => (
    motifSupports(context, cinds, ms, radiusNeighborhood(context, maxDist), neighbors => neighbors.length > minSize)
  ));
}

// Perform differential analysis of motif patterns between two dataset classes using radius-based neighborhoods.
// Returns, for each class, the motifs significantly more frequent in that class.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query_multiple_fov.py#L1206
export async function differentialAnalysisDist(
  multi: SqMultiContext,
  params: DifferentialAnalysisDistParams,
): Promise<Record<string, DifferentialResult[]>> {
  return splitSignificant(await differentialTestsDist(multi, params), params.datasetClasses);
}

