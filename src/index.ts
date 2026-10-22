import Flatbush from 'flatbush';
import { extent } from 'd3-array';
import { assert } from './assert.js';
import { hypergeom, fdrCorrection, roundHalfEven } from './stats.js';
import { mineFrequentPatterns, type FrequentPattern } from './fpgrowth.js';

export { findMaximalPatterns, type FrequentPattern } from './fpgrowth.js';

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
type InitParams = {
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

// Indices of the cells annotated with the cell type ct.
function getCellIndicesOfType(context: SqContext, ct: string): number[] {
  const cinds: number[] = [];
  context.labels.forEach((label, i) => {
    if (label === ct) cinds.push(i);
  });
  if (cinds.length === 0) {
    throw new Error(`Found no ${ct} in the cell type labels!`);
  }
  return cinds;
}

type QueryPoint = {
  x: number,
  y: number,
  // Index of a cell to exclude from the neighborhood (the center cell itself).
  exclude?: number,
};

// Cell type transactions for the radius-based neighborhood of each query point.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_utils.py#L393
function radiusTransactions(context: SqContext, points: QueryPoint[], maxDist: number, minSize: number) {
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
function knnTransactions(context: SqContext, cinds: number[], k: number, maxDist: number) {
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
  const points = cinds.map(i => ({ x: context.xCoords[i], y: context.yCoords[i], exclude: i }));
  const { transactions } = radiusTransactions(context, points, maxDist, minSize);
  return mineFrequentPatterns(transactions, minSupport, ifMax);
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

// Determine significance across all tested motifs, applying FDR correction
// when more than one motif was tested.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/0570c54bd6e046466e05c6b800d6701a7ab15c4a/SpatialQuery/spatial_utils.py#L134
export function applyFdrCorrection(results: MotifEnrichmentResult[], alpha = 0.05): MotifEnrichmentResult[] {
  if (results.length === 0) {
    return [];
  }
  if (results.length === 1) {
    // Single-motif case: significance comes straight from the raw p-value, with no adjusted p-value.
    const { adjPValue: _, ...result } = results[0];
    return [{ ...result, ifSignificant: result.pValue < alpha }];
  }
  const { rejected, adjusted } = fdrCorrection(results.map(r => r.pValue), alpha);
  return results
    .map((r, i) => ({ ...r, adjPValue: adjusted[i], ifSignificant: rejected[i] }))
    .toSorted((a, b) => a.adjPValue - b.adjPValue);
}

export type MotifsParam = string | string[] | string[][];

// Normalize a str / List[str] / List[List[str]] motifs argument into a list of motifs,
// dropping cell types that are not present in the data.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_query.py#L343
function normalizeMotifs(motifs: MotifsParam, labels: string[]): string[][] {
  let motifList: string[][];
  if (typeof motifs === 'string') {
    motifList = [[motifs]];
  } else if (motifs.every(m => typeof m === 'string')) {
    motifList = [motifs as string[]];
  } else {
    motifList = motifs as string[][];
  }
  const labelsUnique = new Set(labels);
  const filteredMotifs: string[][] = [];
  for (const motif of motifList) {
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

// Run the enrichment test for each motif, given a function returning the
// neighborhood (excluding the cell itself) of each cell.
function motifEnrichment(
  context: SqContext,
  ct: string,
  motifs: string[][],
  neighborsOf: (i: number) => number[],
  returnCellIds: boolean,
): MotifEnrichmentResult[] {
  const { labels, obsIndex } = context;
  const numCells = labels.length;
  const cinds = getCellIndicesOfType(context, ct);

  // Encode cell type labels as integers.
  const labelsUnique = Array.from(new Set(labels)).sort();
  const labelToCode = new Map(labelsUnique.map((label, code) => [label, code]));
  const codes = Int32Array.from(labels, label => labelToCode.get(label)!);
  const numTypes = labelsUnique.length;

  // Presence of each cell type in the neighborhood of each cell.
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

  const out: MotifEnrichmentResult[] = [];
  for (const motif of motifs) {
    const motifCodes = Array.from(new Set(motif), m => labelToCode.get(m)!);

    let numMotif = 0;
    for (let i = 0; i < numCells; i++) {
      if (hasMotif(i, motifCodes)) numMotif++;
    }
    const centersWithMotif = cinds.filter(i => hasMotif(i, motifCodes));

    const result = computeMotifEnrichment({
      center: ct,
      motif,
      numCells,
      numCenter: cinds.length,
      numMotif,
      numCenterMotif: centersWithMotif.length,
    });

    if (returnCellIds) {
      // Note: when no center cell has the motif, Python skips the motif entirely
      // (it is dropped from the results). Here we keep it, with empty cell IDs.
      const motifSet = new Set(motif);
      const neighborIndices = new Set<number>();
      for (const i of centersWithMotif) {
        for (const j of centerNeighbors.get(i)!) {
          if (motifSet.has(labels[j])) neighborIndices.add(j);
        }
      }
      result.neighborIds = Array.from(neighborIndices).sort((a, b) => a - b).map(j => obsIndex[j]);
      result.centerIds = centersWithMotif.map(i => obsIndex[i]);
    }
    out.push(result);
  }
  return applyFdrCorrection(out);
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
};

// Perform motif enrichment analysis using k-nearest neighbors.
// Reference: https://spatialquery.readthedocs.io/en/latest/generated/SpatialQuery.spatial_query.motif_enrichment_knn.html
export async function motifEnrichmentKnn(context: SqContext, params: MotifEnrichmentKnnParams): Promise<MotifEnrichmentResult[]> {
  const { ct, k = 30, minSupport = 0.5, maxDist = 20, returnCellIds = false } = params;
  getCellIndicesOfType(context, ct);

  const motifs = params.motifs === undefined
    ? (await findFpKnn(context, { ct, k, minSupport, maxDist })).map(p => p.itemsets)
    : normalizeMotifs(params.motifs, context.labels);
  if (motifs.length === 0) {
    return [];
  }

  const neighborsOf = (i: number) => {
    const { indices, dists } = knnNeighbors(context, i, k);
    return indices.filter((_, n) => dists[n] <= maxDist);
  };
  return motifEnrichment(context, ct, motifs, neighborsOf, returnCellIds);
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
};

// Perform motif enrichment analysis within a radius-based neighborhood.
// Reference: https://spatialquery.readthedocs.io/en/latest/generated/SpatialQuery.spatial_query.motif_enrichment_dist.html
export async function motifEnrichmentDist(context: SqContext, params: MotifEnrichmentDistParams): Promise<MotifEnrichmentResult[]> {
  const { ct, maxDist = 20, minSize = 0, minSupport = 0.5, returnCellIds = false } = params;
  getCellIndicesOfType(context, ct);

  const motifs = params.motifs === undefined
    ? (await findFpDist(context, { ct, maxDist, minSize, minSupport })).map(p => p.itemsets)
    : normalizeMotifs(params.motifs, context.labels);
  if (motifs.length === 0) {
    return [];
  }

  const { xCoords, yCoords } = context;
  const neighborsOf = (i: number) => radiusNeighbors(context, xCoords[i], yCoords[i], maxDist).filter(j => j !== i);
  return motifEnrichment(context, ct, motifs, neighborsOf, returnCellIds);
}
