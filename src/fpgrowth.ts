import { FPGrowth } from 'node-fpgrowth';

export type FrequentPattern = {
  // Sorted list of cell types in the frequent pattern.
  itemsets: string[],
  // Fraction of transactions containing the pattern.
  support: number,
  // Obs IDs of the cells belonging to the pattern (only if returnCellIds is true).
  neighborIds?: string[],
};

// Keep only the maximal patterns: those that are not a proper subset of another pattern.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_utils.py#L68
export function findMaximalPatterns<T extends { itemsets: string[] }>(patterns: T[]): T[] {
  const sets = patterns.map(p => new Set(p.itemsets));
  return patterns.filter((p, i) => !sets.some((other, j) => (
    j !== i
    && other.size > sets[i].size
    && p.itemsets.every(item => other.has(item))
  )));
}

// Mine frequent patterns from transactions of cell type labels.
// Equivalent to the MultiLabelBinarizer + mlxtend fpgrowth + find_maximal_patterns
// steps of build_fptree_dist / build_fptree_knn.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_utils.py#L454
export async function mineFrequentPatterns(
  transactions: string[][],
  minSupport: number,
  ifMax: boolean,
): Promise<FrequentPattern[]> {
  if (transactions.length === 0) {
    return [];
  }
  // Transactions are binarized in Python, so duplicate cell types are collapsed.
  const uniqueTransactions = transactions.map(t => Array.from(new Set(t)));

  const fpgrowth = new FPGrowth<string>(minSupport);
  const itemsets = await fpgrowth.exec(uniqueTransactions);

  // node-fpgrowth reports support as a count; mlxtend reports a fraction.
  const byKey = new Map<string, FrequentPattern>();
  for (const { items, support } of itemsets) {
    const sorted = items.toSorted();
    byKey.set(JSON.stringify(sorted), { itemsets: sorted, support: support / transactions.length });
  }
  let patterns = Array.from(byKey.values());

  if (ifMax) {
    patterns = findMaximalPatterns(patterns);
  }
  // Sort by descending support, breaking ties by itemsets for a deterministic order.
  return patterns.toSorted((a, b) => (
    b.support - a.support || JSON.stringify(a.itemsets).localeCompare(JSON.stringify(b.itemsets))
  ));
}
