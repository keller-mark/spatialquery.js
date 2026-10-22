import Flatbush from 'flatbush';
import { extent } from 'd3-array';
import { assert } from './assert.js';
import { hypergeom, roundHalfEven, applyFdrCorrection } from './stats.js';
import { mineFrequentPatterns, type FrequentPattern } from './fpgrowth.js';
import { nestedMotifDescent } from './nested.js';

export function generateObsIdToObsSetName(obsSetsMembership: Map<string, string[][]>, currSelectedSetGroup: string) {
  // Function that accepts an obsId and returns a single string representing the cell type name.
  return (obsId: string) => {
    const obsIdCellType = obsSetsMembership.get(obsId)
      ?.find(setPath => setPath[0] === currSelectedSetGroup)
      ?.[1]; // Again, we take the second element of the setPath as the cell type name
    assert(typeof obsIdCellType === 'string', "Result of obsIdToObsSetName was falsey");
    return obsIdCellType;
  };
}

type ToObsSetNameFunc = ReturnType<typeof generateObsIdToObsSetName>;

type Coords = Float32Array | Float64Array;

// Reference: https://spatialquery.readthedocs.io/en/latest/generated/SpatialQuery.spatial_query.__init__.html
export type InitParams = {
  obsIndex: string[],
  xCoords: Coords,
  yCoords: Coords,
  obsIdToObsSetName: ToObsSetNameFunc,
  leafSize?: number, // Default: 10
  normalizeSpatialCoord?: boolean, // Default: true
};

export type SqContext = {
  obsIndex: string[],
  // Cell type label of each cell, aligned with obsIndex.
  labels: string[],
  // Spatial coordinates, normalized if normalizeSpatialCoord was true.
  xCoords: Coords,
  yCoords: Coords,
  flatbushIndex: Flatbush,
  xExtent: [number, number],
  yExtent: [number, number],
}

// Scale spatial coordinates so that the mean nearest neighbor distance equals 1.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/0570c54bd6e046466e05c6b800d6701a7ab15c4a/SpatialQuery/spatial_utils.py#L804
export function normalizeSpatialCoords(xCoords: Coords, yCoords: Coords) {
  const numCells = xCoords.length;
  assert(numCells >= 2, "At least two cells are required to normalize spatial coordinates.");

  // Temporary spatial index over the original coordinates, used to find each cell's nearest neighbor.
  const index = new Flatbush(numCells);
  for (let i = 0; i < numCells; i++) {
    index.add(xCoords[i], yCoords[i]);
  }
  index.finish();

  let nnDistSum = 0;
  for (let i = 0; i < numCells; i++) {
    // Exclude the cell itself (equivalent to taking the second result of a k=2 KDTree query).
    const [j] = index.neighbors(xCoords[i], yCoords[i], 1, Infinity, (j) => j !== i);
    nnDistSum += Math.hypot(xCoords[i] - xCoords[j], yCoords[i] - yCoords[j]);
  }
  const nnDist = nnDistSum / numCells;
  assert(nnDist > 0, "Mean nearest neighbor distance was zero; cannot normalize spatial coordinates.");

  const scaleFactor = 1.0 / nnDist;
  return {
    xCoords: Float64Array.from(xCoords, x => x * scaleFactor),
    yCoords: Float64Array.from(yCoords, y => y * scaleFactor),
    scaleFactor,
  };
}

// Build the context object.
export function buildContext(params: InitParams): SqContext {
  const {
    obsIndex, // Array of each obs ID, as a string.
    xCoords: rawXCoords, // TypedArray containing X coordinate of each cell (centroid)
    yCoords: rawYCoords, // TypedArray containing Y coordinate of each cell (centroid)
    obsIdToObsSetName, // Function mapping an obs ID to its cell type label.

    leafSize = 10, // Leaf size parameter for KDTree construction. Larger values reduce tree depth but increase computation per query.
    normalizeSpatialCoord = true, // If True, normalizes spatial coordinates so mean nearest neighbor distance equals 1. This makes max_dist parameter interpretable as “number of cell diameters.” Set to False if original spatial units (e.g., micrometers) are required and should be preserved for downstream analysis.
  } = params;

  const { xCoords, yCoords } = normalizeSpatialCoord
    ? normalizeSpatialCoords(rawXCoords, rawYCoords)
    : { xCoords: rawXCoords, yCoords: rawYCoords };

  const xExtent = extent(xCoords);
  const yExtent = extent(yCoords);

  assert(typeof xExtent[0] === 'number' && typeof xExtent[1] === 'number', "X extent min or max was undefined.");
  assert(typeof yExtent[0] === 'number' && typeof yExtent[1] === 'number', "Y extent min or max was undefined.");

  // Initialize Flatbush for N items
  const flatbushIndex = new Flatbush(obsIndex.length, leafSize);

  for (let i = 0; i < obsIndex.length; i++) {
    flatbushIndex.add(xCoords[i], yCoords[i]);
  }

  // perform the indexing
  flatbushIndex.finish();

  return {
    obsIndex,
    labels: obsIndex.map(obsId => obsIdToObsSetName(obsId)),
    xCoords,
    yCoords,
    xExtent,
    yExtent,
    flatbushIndex,
  };
}

// ---------------------------------------------------------------------------
// Neighborhood queries
// ---------------------------------------------------------------------------

// Indices of cells within maxDist (inclusive) of a point, like KDTree.query_ball_point.
function radiusNeighbors(context: SqContext, x: number, y: number, maxDist: number): number[] {
  return context.flatbushIndex.neighbors(x, y, Infinity, maxDist);
}

// The k nearest neighbors of cell i (excluding itself) and their distances,
// like KDTree.query(k=k+1) with the first result (the cell itself) dropped.
function knnNeighbors(context: SqContext, i: number, k: number) {
  const { xCoords, yCoords, flatbushIndex } = context;
  const x = xCoords[i];
  const y = yCoords[i];
  const indices = flatbushIndex.neighbors(x, y, k, Infinity, j => j !== i);
  const dists = indices.map(j => Math.hypot(xCoords[j] - x, yCoords[j] - y));
  return { indices, dists };
}

// Indices of the cells annotated with the cell type ct (possibly none).
export function findCellIndicesOfType(context: SqContext, ct: string): number[] {
  const cinds: number[] = [];
  context.labels.forEach((label, i) => {
    if (label === ct) cinds.push(i);
  });
  return cinds;
}

// Indices of the cells annotated with the cell type ct, throwing if there are none.
function getCellIndicesOfType(context: SqContext, ct: string): number[] {
  const cinds = findCellIndicesOfType(context, ct);
  if (cinds.length === 0) {
    throw new Error(`Found no ${ct} in the cell type labels!`);
  }
  return cinds;
}

export type QueryPoint = {
  x: number,
  y: number,
  // Index of a cell to exclude from the neighborhood (the center cell itself).
  exclude?: number,
};

// Neighborhood of each cell (excluding the cell itself) used by the k-NN motif enrichment and composition:
// the k nearest neighbors within maxDist (inclusive).
export function knnNeighborhood(context: SqContext, k: number, maxDist: number) {
  return (i: number) => {
    const { indices, dists } = knnNeighbors(context, i, k);
    return indices.filter((_, n) => dists[n] <= maxDist);
  };
}

// Neighborhood of each cell (excluding the cell itself) used by the radius-based motif enrichment and composition.
export function radiusNeighborhood(context: SqContext, maxDist: number) {
  const { xCoords, yCoords } = context;
  return (i: number) => radiusNeighbors(context, xCoords[i], yCoords[i], maxDist).filter(j => j !== i);
}

// Cell type transactions for the radius-based neighborhood of each query point.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_utils.py#L393
export function radiusTransactions(context: SqContext, points: QueryPoint[], maxDist: number, minSize: number) {
  const transactions: string[][] = [];
  const neighborLists: number[][] = [];
  for (const { x, y, exclude } of points) {
    const idx = radiusNeighbors(context, x, y, maxDist);
    if (idx.length === 0) continue;
    const validIndices = idx.filter(j => j !== exclude);
    // Python compares the number of neighbor cells (not unique cell types) against min_size.
    if (validIndices.length > minSize) {
      transactions.push(validIndices.map(j => context.labels[j]));
      neighborLists.push(validIndices);
    }
  }
  return { transactions, neighborLists };
}

// Cell type transactions for the k-NN neighborhood of each center cell.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_utils.py#L475
export function knnTransactions(context: SqContext, cinds: number[], k: number, maxDist: number) {
  const transactions: string[][] = [];
  // In Python, the center cell itself (at distance 0) must pass the dist < max_dist filter,
  // otherwise the neighborhood is skipped entirely. Neighborhoods may otherwise be empty.
  if (!(maxDist > 0)) {
    return transactions;
  }
  for (const i of cinds) {
    const { indices, dists } = knnNeighbors(context, i, k);
    // Note: strict inequality here, unlike in motif_enrichment_knn.
    transactions.push(indices.filter((_, n) => dists[n] < maxDist).map(j => context.labels[j]));
  }
  return transactions;
}

// For each pattern, collect the cells (with cell types in the pattern) from the
// neighborhoods that contain the full pattern.
function attachNeighborIds(
  context: SqContext,
  patterns: FrequentPattern[],
  transactions: string[][],
  neighborLists: number[][],
): FrequentPattern[] {
  const transactionSets = transactions.map(t => new Set(t));
  return patterns.map((pattern) => {
    const motif = new Set(pattern.itemsets);
    const ids = new Set<number>();
    transactionSets.forEach((transaction, t) => {
      if (pattern.itemsets.every(ct => transaction.has(ct))) {
        for (const j of neighborLists[t]) {
          if (motif.has(context.labels[j])) ids.add(j);
        }
      }
    });
    const neighborIds = Array.from(ids).sort((a, b) => a - b).map(j => context.obsIndex[j]);
    return { ...pattern, neighborIds };
  });
}

// ---------------------------------------------------------------------------
// find_patterns_grid / find_patterns_rand
// ---------------------------------------------------------------------------

type FindPatternsParams = {
  // Maximum distance to consider a cell as a neighbor.
  maxDist?: number, // Default: 20
  // Minimum neighborhood size for each point to consider.
  minSize?: number, // Default: 0
  // Threshold of frequency to consider a pattern as a frequent pattern.
  minSupport?: number, // Default: 0.5
  // Whether to return the obs IDs of the cells belonging to each pattern.
  returnCellIds?: boolean, // Default: false
};

async function findPatternsAtPoints(
  context: SqContext,
  points: [number, number][],
  params: FindPatternsParams,
): Promise<FrequentPattern[]> {
  const {
    maxDist = 20,
    minSize = 0,
    minSupport = 0.5,
    returnCellIds = false,
  } = params;

  // Note: Python (build_fptree_dist with cinds=None) also excludes the cell whose index
  // equals the index of the query point. We do not reproduce that quirk.
  const queryPoints = points.map(([x, y]) => ({ x, y }));
  const { transactions, neighborLists } = radiusTransactions(context, queryPoints, maxDist, minSize);
  const patterns = await mineFrequentPatterns(transactions, minSupport, true);

  if (returnCellIds) {
    return attachNeighborIds(context, patterns, transactions, neighborLists);
  }
  return patterns;
}

// Equivalent to np.arange(start, stop, step).
function arange(start: number, stop: number, step: number): number[] {
  const n = Math.max(0, Math.ceil((stop - start) / step));
  return Array.from({ length: n }, (_, i) => start + i * step);
}

// Grid points used by findPatternsGrid, with spacing maxDist.
// Equivalent to the grid returned by find_patterns_grid(return_grid=True).
export function buildGridPoints(context: SqContext, maxDist = 20): [number, number][] {
  const { xExtent, yExtent } = context;
  const xGrid = arange(xExtent[0] - maxDist, xExtent[1] + maxDist, maxDist);
  const yGrid = arange(yExtent[0] - maxDist, yExtent[1] + maxDist, maxDist);
  const gridPoints: [number, number][] = [];
  for (const x of xGrid) {
    for (const y of yGrid) {
      gridPoints.push([x, y]);
    }
  }
  return gridPoints;
}

// Create a grid and use it to find surrounding patterns in spatial data.
// Reference: https://spatialquery.readthedocs.io/en/latest/generated/SpatialQuery.spatial_query.find_patterns_grid.html
export async function findPatternsGrid(context: SqContext, params: FindPatternsParams = {}) {
  const { maxDist = 20 } = params;
  return findPatternsAtPoints(context, buildGridPoints(context, maxDist), params);
}

// Randomly generate points and use them to find surrounding patterns in spatial data.
// Uses Math.random, so results will not exactly match Python (which uses a seeded numpy RNG).
// Reference: https://spatialquery.readthedocs.io/en/latest/generated/SpatialQuery.spatial_query.find_patterns_rand.html
export async function findPatternsRand(
  context: SqContext,
  params: FindPatternsParams & {
    // Number of random points to generate.
    nPoints?: number, // Default: 1000
  } = {},
) {
  const { nPoints = 1000 } = params;
  const { xExtent: [xMin, xMax], yExtent: [yMin, yMax] } = context;
  // As in Python, all x coordinates are drawn before all y coordinates.
  const xs = Array.from({ length: nPoints }, () => Math.random() * (xMax - xMin) + xMin);
  const ys = Array.from({ length: nPoints }, () => Math.random() * (yMax - yMin) + yMin);
  const points = xs.map((x, i) => [x, ys[i]] as [number, number]);
  return findPatternsAtPoints(context, points, params);
}

// ---------------------------------------------------------------------------
// find_fp_knn / find_fp_dist
// ---------------------------------------------------------------------------

type FindFpKnnParams = {
  // Cell type of the center cells.
  ct: string,
  // Number of nearest neighbors.
  k?: number, // Default: 30
  // Threshold of frequency to consider a pattern as a frequent pattern.
  minSupport?: number, // Default: 0.5
  // Maximum distance for considering a cell as a neighbor.
  maxDist?: number, // Default: 20
  // Return only maximal patterns.
  ifMax?: boolean, // Default: true
};

// Find frequent patterns within the k-NN neighborhoods of a cell type.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query.py#L155
export async function findFpKnn(context: SqContext, params: FindFpKnnParams): Promise<FrequentPattern[]> {
  const { ct, k = 30, minSupport = 0.5, maxDist = 20, ifMax = true } = params;
  const cinds = getCellIndicesOfType(context, ct);
  const transactions = knnTransactions(context, cinds, k, maxDist);
  return mineFrequentPatterns(transactions, minSupport, ifMax);
}

type FindFpDistParams = {
  // Cell type of the center cells.
  ct: string,
  // Maximum distance for considering a cell as a neighbor.
  maxDist?: number, // Default: 20
  // Minimum neighborhood size for each center cell to consider.
  minSize?: number, // Default: 0
  // Threshold of frequency to consider a pattern as a frequent pattern.
  minSupport?: number, // Default: 0.5
  // Return only maximal patterns.
  ifMax?: boolean, // Default: true
};

// Find frequent patterns within the radius-based neighborhoods of a cell type.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query.py#L202
export async function findFpDist(context: SqContext, params: FindFpDistParams): Promise<FrequentPattern[]> {
  const { ct, maxDist = 20, minSize = 0, minSupport = 0.5, ifMax = true } = params;
  const cinds = getCellIndicesOfType(context, ct);
  const { transactions } = radiusTransactions(context, centerQueryPoints(context, cinds), maxDist, minSize);
  return mineFrequentPatterns(transactions, minSupport, ifMax);
}

// Query points at the given cells, excluding each cell from its own neighborhood.
export function centerQueryPoints(context: SqContext, cinds: number[]): QueryPoint[] {
  return cinds.map(i => ({ x: context.xCoords[i], y: context.yCoords[i], exclude: i }));
}

// ---------------------------------------------------------------------------
// Motif enrichment
// ---------------------------------------------------------------------------

type MotifEnrichmentCountsParams = {
  // Cell type of interest (the center cell type).
  center: string,
  // List of cell types in the motif.
  motif: string[],
  // Total number of cells (all cell types).
  numCells: number,
  // Number of cells annotated as the center cell type.
  numCenter: number,
  // Number of cells (of any cell type) whose k-NN or radius neighborhood contains the motif.
  numMotif: number,
  // Number of center cells whose k-NN or radius neighborhood contains the motif.
  numCenterMotif: number,
};

export type MotifEnrichmentResult = {
  center: string,
  motifs: string[],
  nCenterMotif: number,
  nCenter: number,
  nMotif: number,
  expectation: number,
  pValue: number,
  ifSignificant?: boolean,
  adjPValue?: number,
  // Obs IDs of neighbor cells with motif cell types (only if returnCellIds is true).
  neighborIds?: string[],
  // Obs IDs of center cells with the motif in their neighborhood (only if returnCellIds is true).
  centerIds?: string[],
};

// Compute the hypergeometric enrichment test for a single motif,
// given the neighborhood counts from a k-NN or radius-based query.
// Currently, SpatialQuery only provides p-values for cellTypeOfInterest queries.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/0570c54bd6e046466e05c6b800d6701a7ab15c4a/SpatialQuery/spatial_query.py#L490C13-L492C131
export function computeMotifEnrichment(params: MotifEnrichmentCountsParams): MotifEnrichmentResult {
  const {
    center,
    motif,
    numCells,
    numCenter,
    numMotif,
    numCenterMotif,
  } = params;

  // If the center cell type appears in the motif, divide by the number of times it appears.
  // Python's round() rounds half to even, so we match that here.
  let nCenter = numCenter;
  const centerCount = motif.filter(ct => ct === center).length;
  if (centerCount > 0) {
    nCenter = roundHalfEven(nCenter / centerCount);
  }

  // M = total number of cells, n = number of "successes" (center cells),
  // N = number drawn without replacement (cells with the motif in their neighborhood).
  const hyge = hypergeom(numCells, nCenter, numMotif);

  return {
    center,
    motifs: motif.toSorted(),
    nCenterMotif: numCenterMotif,
    nCenter,
    nMotif: numMotif,
    expectation: hyge.mean(),
    pValue: hyge.sf(numCenterMotif),
  };
}

export type MotifsParam = string | string[] | string[][];

// Turn a str / List[str] / List[List[str]] motifs argument into a list of motifs.
function toMotifList(motifs: MotifsParam): string[][] {
  if (typeof motifs === 'string') {
    return [[motifs]];
  }
  if (motifs.every(m => typeof m === 'string')) {
    return [motifs as string[]];
  }
  return motifs as string[][];
}

// Normalize a motifs argument into a list of motifs, dropping cell types that are not
// present in the data. Used by the motif enrichment functions (which keep the motif order and duplicates).
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query.py#L343
export function normalizeMotifs(motifs: MotifsParam, labelsUnique: Set<string>): string[][] {
  const filteredMotifs: string[][] = [];
  for (const motif of toMotifList(motifs)) {
    const motifExc = motif.filter(m => !labelsUnique.has(m));
    if (motifExc.length > 0) {
      console.warn(`Found no ${JSON.stringify(motifExc)} in the cell type labels. Ignoring them.`);
    }
    const validMotif = motif.filter(m => labelsUnique.has(m));
    if (validMotif.length > 0) {
      filteredMotifs.push(validMotif);
    }
  }
  return filteredMotifs;
}

// Normalize a motifs argument into sorted, de-duplicated motifs, dropping cell types that are
// not present in the data. Used by the motif composition functions.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_utils.py#L300
export function normalizeMotifList(motifs: MotifsParam, labelsUnique: Set<string>): string[][] {
  return normalizeMotifs(motifs, labelsUnique).map(motif => Array.from(new Set(motif)).sort());
}

// Neighborhood counts for the motif enrichment test, within one FOV.
export type MotifCounts = {
  numCells: number,
  numCenter: number,
  numMotif: number,
  numCenterMotif: number,
  // Indices of the center cells with the motif in their neighborhood (only if returnCellIds is true).
  centerIndices?: number[],
  // Indices of the motif-type neighbors of those center cells (only if returnCellIds is true).
  neighborIndices?: number[],
};

// Count, for each motif, the cells (and center cells) whose neighborhood contains every
// cell type of the motif, given a function returning the neighborhood of each cell.
// The center cell type may be absent (relevant for multi-FOV).
export function countMotifs(
  context: SqContext,
  ct: string,
  motifs: string[][],
  neighborsOf: (i: number) => number[],
  returnCellIds: boolean,
): MotifCounts[] {
  const { labels } = context;
  const numCells = labels.length;
  const cinds = findCellIndicesOfType(context, ct);

  // Encode cell type labels as integers.
  const labelsUnique = Array.from(new Set(labels)).sort();
  const labelToCode = new Map(labelsUnique.map((label, code) => [label, code]));
  const numTypes = labelsUnique.length;

  // A motif with a cell type missing from this FOV cannot be present in any neighborhood.
  const isActive = motifs.map(motif => motif.every(m => labelToCode.has(m)));
  const emptyCounts = (): MotifCounts => ({
    numCells,
    numCenter: cinds.length,
    numMotif: 0,
    numCenterMotif: 0,
    ...(returnCellIds ? { centerIndices: [], neighborIndices: [] } : {}),
  });
  if (!isActive.some(Boolean)) {
    return motifs.map(emptyCounts);
  }

  // Presence of each cell type in the neighborhood of each cell.
  const codes = Int32Array.from(labels, label => labelToCode.get(label)!);
  const presence = new Uint8Array(numCells * numTypes);
  const centerNeighbors = new Map<number, number[]>();
  const isCenter = new Uint8Array(numCells);
  for (const i of cinds) isCenter[i] = 1;
  for (let i = 0; i < numCells; i++) {
    const neighbors = neighborsOf(i);
    for (const j of neighbors) {
      presence[i * numTypes + codes[j]] = 1;
    }
    if (returnCellIds && isCenter[i]) {
      centerNeighbors.set(i, neighbors);
    }
  }

  const hasMotif = (i: number, motifCodes: number[]) => motifCodes.every(c => presence[i * numTypes + c] === 1);

  return motifs.map((motif, m) => {
    if (!isActive[m]) {
      return emptyCounts();
    }
    const motifCodes = Array.from(new Set(motif), t => labelToCode.get(t)!);

    let numMotif = 0;
    for (let i = 0; i < numCells; i++) {
      if (hasMotif(i, motifCodes)) numMotif++;
    }
    const centersWithMotif = cinds.filter(i => hasMotif(i, motifCodes));
    const counts: MotifCounts = {
      numCells,
      numCenter: cinds.length,
      numMotif,
      numCenterMotif: centersWithMotif.length,
    };

    if (returnCellIds) {
      const motifSet = new Set(motif);
      const neighborIndices = new Set<number>();
      for (const i of centersWithMotif) {
        for (const j of centerNeighbors.get(i)!) {
          if (motifSet.has(labels[j])) neighborIndices.add(j);
        }
      }
      counts.centerIndices = centersWithMotif;
      counts.neighborIndices = Array.from(neighborIndices).sort((a, b) => a - b);
    }
    return counts;
  });
}

// Run the enrichment test for each motif, given a function returning the
// neighborhood (excluding the cell itself) of each cell.
function motifEnrichment(
  context: SqContext,
  ct: string,
  motifs: string[][],
  neighborsOf: (i: number) => number[],
  returnCellIds: boolean,
): MotifEnrichmentResult[] {
  const { obsIndex } = context;
  const counts = countMotifs(context, ct, motifs, neighborsOf, returnCellIds);
  const out = motifs.map((motif, m) => {
    const { numCells, numCenter, numMotif, numCenterMotif, centerIndices, neighborIndices } = counts[m];
    const result = computeMotifEnrichment({ center: ct, motif, numCells, numCenter, numMotif, numCenterMotif });
    if (returnCellIds) {
      // Note: when no center cell has the motif, Python skips the motif entirely
      // (it is dropped from the results). Here we keep it, with empty cell IDs.
      result.neighborIds = neighborIndices!.map(j => obsIndex[j]);
      result.centerIds = centerIndices!.map(i => obsIndex[i]);
    }
    return result;
  });
  return applyFdrCorrection(out);
}

// How the motifs to test are chosen when motifs are not provided.
// 'maximal': test the maximal frequent patterns only.
// 'nested': test the maximal patterns, then descend into the sub-patterns of any that are
// not significant (see nestedMotifDescent).
export type EnrichmentMode = 'maximal' | 'nested';

export function assertEnrichmentMode(mode: string) {
  if (mode !== 'maximal' && mode !== 'nested') {
    throw new Error(`mode must be 'nested' or 'maximal', got '${mode}'`);
  }
}

type MotifEnrichmentKnnParams = {
  // Cell type of the center cells.
  ct: string,
  // Motifs to test. If not provided, the frequent patterns within the
  // neighborhoods of the center cell type are used as motifs.
  motifs?: MotifsParam,
  // Number of nearest neighbors.
  k?: number, // Default: 30
  // Threshold of frequency to consider a pattern as a frequent pattern.
  minSupport?: number, // Default: 0.5
  // Maximum distance for neighbors.
  maxDist?: number, // Default: 20
  // Whether to return the obs IDs of center and neighbor cells for each motif.
  returnCellIds?: boolean, // Default: false
  // How the motifs to test are chosen when motifs is not provided.
  mode?: EnrichmentMode, // Default: 'maximal'
};

// Perform motif enrichment analysis using k-nearest neighbors.
// Reference: https://spatialquery.readthedocs.io/en/latest/generated/SpatialQuery.spatial_query.motif_enrichment_knn.html
export async function motifEnrichmentKnn(context: SqContext, params: MotifEnrichmentKnnParams): Promise<MotifEnrichmentResult[]> {
  const { ct, k = 30, minSupport = 0.5, maxDist = 20, returnCellIds = false, mode = 'maximal' } = params;
  getCellIndicesOfType(context, ct);
  assertEnrichmentMode(mode);

  if (mode === 'nested' && params.motifs === undefined) {
    return nestedMotifDescent({
      testMotifs: motifs => motifEnrichmentKnn(context, { ...params, motifs, mode: 'maximal' }),
      frequentPatterns: await findFpKnn(context, { ct, k, minSupport, maxDist, ifMax: false }),
    });
  }

  const motifs = params.motifs === undefined
    ? (await findFpKnn(context, { ct, k, minSupport, maxDist })).map(p => p.itemsets)
    : normalizeMotifs(params.motifs, new Set(context.labels));
  if (motifs.length === 0) {
    return [];
  }
  return motifEnrichment(context, ct, motifs, knnNeighborhood(context, k, maxDist), returnCellIds);
}

type MotifEnrichmentDistParams = {
  // Cell type of the center cells.
  ct: string,
  // Motifs to test. If not provided, the frequent patterns within the
  // neighborhoods of the center cell type are used as motifs.
  motifs?: MotifsParam,
  // Maximum distance for considering a cell as a neighbor.
  maxDist?: number, // Default: 20
  // Minimum neighborhood size for each point to consider (only used for finding frequent patterns).
  minSize?: number, // Default: 0
  // Threshold of frequency to consider a pattern as a frequent pattern.
  minSupport?: number, // Default: 0.5
  // Whether to return the obs IDs of center and neighbor cells for each motif.
  returnCellIds?: boolean, // Default: false
  // How the motifs to test are chosen when motifs is not provided.
  mode?: EnrichmentMode, // Default: 'maximal'
};

// Perform motif enrichment analysis within a radius-based neighborhood.
// Reference: https://spatialquery.readthedocs.io/en/latest/generated/SpatialQuery.spatial_query.motif_enrichment_dist.html
export async function motifEnrichmentDist(context: SqContext, params: MotifEnrichmentDistParams): Promise<MotifEnrichmentResult[]> {
  const { ct, maxDist = 20, minSize = 0, minSupport = 0.5, returnCellIds = false, mode = 'maximal' } = params;
  getCellIndicesOfType(context, ct);
  assertEnrichmentMode(mode);

  if (mode === 'nested' && params.motifs === undefined) {
    return nestedMotifDescent({
      testMotifs: motifs => motifEnrichmentDist(context, { ...params, motifs, mode: 'maximal' }),
      frequentPatterns: await findFpDist(context, { ct, maxDist, minSize, minSupport, ifMax: false }),
    });
  }

  const motifs = params.motifs === undefined
    ? (await findFpDist(context, { ct, maxDist, minSize, minSupport })).map(p => p.itemsets)
    : normalizeMotifs(params.motifs, new Set(context.labels));
  if (motifs.length === 0) {
    return [];
  }
  return motifEnrichment(context, ct, motifs, radiusNeighborhood(context, maxDist), returnCellIds);
}

// ---------------------------------------------------------------------------
// Motif composition
// ---------------------------------------------------------------------------

export type MotifCompositionRow = {
  // Center cell type.
  center: string,
  // Sorted list of cell types in the motif.
  motifs: string[],
  // Motif cell type this row describes.
  cellType: string,
  // Number of cells of cellType in the neighborhood.
  nCells: number,
  // Number of motif-type cells in the neighborhood.
  nMotifCells: number,
  // nCells / nMotifCells
  proportion: number,
};

// Long-format composition of the motif-type cells in the neighborhoods of the center cells
// (cinds) whose neighborhood contains every motif cell type: one row per (center cell, motif cell type).
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_utils.py#L362
export function motifCompositionRows(
  context: SqContext,
  ct: string,
  motifs: string[][],
  cinds: number[],
  neighborsOf: (i: number) => number[],
): MotifCompositionRow[] {
  const { labels } = context;
  // Per-cell-type neighbor counts of each center cell.
  const centerCounts = cinds.map((i) => {
    const counts = new Map<string, number>();
    for (const j of neighborsOf(i)) {
      counts.set(labels[j], (counts.get(labels[j]) ?? 0) + 1);
    }
    return counts;
  });

  const rows: MotifCompositionRow[] = [];
  for (const motif of motifs) {
    for (const counts of centerCounts) {
      const motifCounts = motif.map(t => counts.get(t) ?? 0);
      if (!motifCounts.every(c => c > 0)) continue;
      const total = motifCounts.reduce((a, b) => a + b, 0);
      motif.forEach((cellType, n) => {
        rows.push({
          center: ct,
          motifs: motif,
          cellType,
          nCells: motifCounts[n],
          nMotifCells: total,
          proportion: motifCounts[n] / total,
        });
      });
    }
  }
  return rows;
}

type MotifCompositionKnnParams = {
  // Cell type of the center cells.
  ct: string,
  // Motifs to analyze, e.g. the motifs of the motifEnrichmentKnn output.
  motifs: MotifsParam,
  // Number of nearest neighbors.
  k?: number, // Default: 30
  // Maximum distance for neighbors.
  maxDist?: number, // Default: 20
};

// Compute the composition of motif cell types in the k-NN neighborhoods of center cells.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query.py#L681
export function motifCompositionKnn(context: SqContext, params: MotifCompositionKnnParams): MotifCompositionRow[] {
  const { ct, k = 30, maxDist = 20 } = params;
  const cinds = getCellIndicesOfType(context, ct);
  const motifs = normalizeMotifList(params.motifs, new Set(context.labels));
  return motifCompositionRows(context, ct, motifs, cinds, knnNeighborhood(context, k, maxDist));
}

type MotifCompositionDistParams = {
  // Cell type of the center cells.
  ct: string,
  // Motifs to analyze, e.g. the motifs of the motifEnrichmentDist output.
  motifs: MotifsParam,
  // Maximum distance for considering a cell as a neighbor.
  maxDist?: number, // Default: 20
};

// Compute the composition of motif cell types in the radius-based neighborhoods of center cells.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query.py#L631
export function motifCompositionDist(context: SqContext, params: MotifCompositionDistParams): MotifCompositionRow[] {
  const { ct, maxDist = 20 } = params;
  const cinds = getCellIndicesOfType(context, ct);
  const motifs = normalizeMotifList(params.motifs, new Set(context.labels));
  return motifCompositionRows(context, ct, motifs, cinds, radiusNeighborhood(context, maxDist));
}

