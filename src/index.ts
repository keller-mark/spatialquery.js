import Flatbush from 'flatbush';
import { FPGrowth, Itemset } from 'node-fpgrowth';
import { extent } from 'd3-array';
import { InternMap } from 'internmap';
import { assert } from './assert.js';
import { hypergeom, fdrCorrection, roundHalfEven } from './stats.js';

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

// Reference: https://spatialquery.readthedocs.io/en/latest/generated/SpatialQuery.spatial_query.__init__.html
type InitParams = {
  obsIndex: string[],
  xCoords: Float32Array,
  yCoords: Float32Array,
  obsIdToObsSetName: ToObsSetNameFunc,
  leafSize?: number, // Default: 10
  normalizeSpatialCoord?: boolean, // Default: true
};

type SqContext = {
  obsIndex: string[],
  flatbushIndex: Flatbush,
  xExtent: [number, number],
  yExtent: [number, number],
}

// Build the context object.
export function buildContext(params: InitParams): SqContext {
  const {
    obsIndex, // Array of each obs ID, as a string.
    xCoords, // TypedArray containing X coordinate of each cell (centroid)
    yCoords, // TypedArray containing Y coordinate of each cell (centroid)

    leafSize = 10, // Leaf size parameter for KDTree construction. Larger values reduce tree depth but increase computation per query.
    normalizeSpatialCoord = true, // If True, normalizes spatial coordinates so mean nearest neighbor distance equals 1. This makes max_dist parameter interpretable as “number of cell diameters.” Set to False if original spatial units (e.g., micrometers) are required and should be preserved for downstream analysis.
  } = params;

  // TODO: implement normalizeSpatialCoord here.

  // TODO: ensure that xExtent and yExtent reflect post-normalizeSpatialCoord handling.
  const xExtent = extent(xCoords);
  const yExtent = extent(yCoords);

  assert(typeof xExtent[0] === 'number' && typeof xExtent[1] === 'number', "X extent min or max was undefined.");
  assert(typeof yExtent[0] === 'number' && typeof yExtent[1] === 'number', "Y extent min or max was undefined.");

  // Initialize Flatbush for N items
  const flatbushIndex = new Flatbush(obsIndex.length, leafSize);

  // fill it with 1000 rectangles
  for (let i = 0; i < obsIndex.length; i++) {
    // TODO: use normalized values here if normalizeSpatialCoord is true.
    flatbushIndex.add(xCoords[i], yCoords[i]);
  }

  // perform the indexing
  flatbushIndex.finish();

  return {
    obsIndex,
    xExtent,
    yExtent,
    flatbushIndex,
  };
}

// Reference: https://spatialquery.readthedocs.io/en/latest/generated/SpatialQuery.spatial_query.find_patterns_grid.html
type FindPatternsGridParams = {
  // Threshold of frequency to consider a pattern as a frequent pattern.
  minSupport?: number, // Default: 0.5
  // Maximum distance to consider a cell as a neighbor. Also used as grid spacing.
  maxDist?: number, // Default: 20
  // Minimum neighborhood size for each grid point to consider.
  minSize?: number, // Default: 0
};

export async function findPatternsGrid(
  context: SqContext,
  params: FindPatternsGridParams,
  obsIdToObsSetName: ToObsSetNameFunc,
) {
  const {
    obsIndex,
    flatbushIndex,
    xExtent,
    yExtent,
  } = context;
  const {
    minSupport = 0.5,
    maxDist = 20,
    minSize = 0,
  } = params;

  // Obtain [x,y] coordinates for the grid points.
  const gridPoints = [];
  for(let gridX = xExtent[0] - maxDist; gridX < xExtent[1] + maxDist; gridX += maxDist) {
    for(let gridY = yExtent[0] - maxDist; gridY < yExtent[1] + maxDist; gridY += maxDist) {
      gridPoints.push([gridX, gridY]);
    }
  }

  // One list of observation IDs per grid point query
  const indexLists = [];
  for(const gridPoint of gridPoints) {
    // TODO: also support k-nearest-neighbor query.
    // make a k-nearest-neighbors query
    // const neighborIds = flatbushIndex.neighbors(xCoords[10], yCoords[10], 5);

    // make a bounding box query
    const [gridX, gridY] = gridPoint;
    const minX = gridX - maxDist/2;
    const maxX = gridX + maxDist/2;
    const minY = gridY - maxDist/2;
    const maxY = gridY + maxDist/2;
    // TODO: filter to a circular region? (rather than a rectangular one)
    const found = flatbushIndex.search(minX, minY, maxX, maxY).map((i) => obsIndex[i]);
    indexLists.push(found);
  }

  // Keep track of the obsIds that match the transactions.
  // This is a map from transaction to a list of observation indices.
  // We use InternMap to allow using arrays as keys.
  const transactionToObsIdsMap = new InternMap([], JSON.stringify) as Map<string[], string[]>;

  // Convert lists of observation indices to transactions.
  const transactions = [];
  for(const indexList of indexLists) {
    if(indexList.length > 0) {
      // For each indexList, we create a transaction of cell types.
      // We use the obsSetsMembership to get the cell types for each observation ID.
      const cellTypes = indexList.map(obsId => obsIdToObsSetName(obsId));
      const transaction = Array.from(new Set(cellTypes)).toSorted();
      if(transaction.length >= minSize) {
        transactions.push(transaction);
        // Store the transaction to observation IDs mapping.
        // Reference: https://github.com/ShaokunAn/Spatial-Query/blob/0570c54bd6e046466e05c6b800d6701a7ab15c4a/SpatialQuery/spatial_query.py#L789
        if (!transactionToObsIdsMap.has(transaction)) {
          transactionToObsIdsMap.set(transaction, indexList);
        } else {
          const prevTransaction = transactionToObsIdsMap.get(transaction)!;
          transactionToObsIdsMap.set(transaction, prevTransaction.concat(indexList));
        }
      }
    }
  }

  const fpgrowth = new FPGrowth<string>(minSupport);
  const itemsets = await fpgrowth.exec(transactions);

  // Returns an array representing the frequent itemsets.
  const validItemsets = itemsets.filter(d => d.items.length >= minSize);
  const sortedItemsets = validItemsets.toSorted((a, b) => b.support - a.support);

  const sortedItemsetsWithObsIds = sortedItemsets.map((itemset) => {
    // For each itemset, get the observation IDs that match the transaction.
    const motif = itemset.items;
    let matchingObsIds: string[] = [];
    for(const transaction of transactionToObsIdsMap.keys()) {
      // Check if all motif items are present in the transaction.
      if (motif.every(item => transaction.includes(item))) {
        // If so, we can use this transaction to get the observation IDs.
        const obsIds = transactionToObsIdsMap.get(transaction)!;
        matchingObsIds = matchingObsIds.concat(obsIds);
        // Filter the matchingObsIds to only include those obsIds corresponding to
        // cell types in the current motif.
        matchingObsIds = matchingObsIds.filter(obsId => {
          const obsIdCellType = obsIdToObsSetName(obsId);
          return motif.includes(obsIdCellType);
        });
      }
    }
    const obsIds = Array.from(new Set(matchingObsIds));
    return { ...itemset, obsIds };
  });
  console.log(sortedItemsetsWithObsIds);

  return sortedItemsets;
}





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
