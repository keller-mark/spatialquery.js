import { applyFdrCorrection } from './stats.js';
import { findMaximalPatterns, type FrequentPattern } from './fpgrowth.js';

// Nested motif testing.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_utils.py#L97
//
// The maximal-pattern reduction happens BEFORE the hypergeometric test, so a sub-pattern
// of a non-significant maximal pattern never enters the tested set at all. When a maximal
// pattern is diluted by one abundant, unrelated cell type, the whole pattern fails the
// test and the informative core inside it is never examined.
//
// Nested testing walks into those patterns:
//
//   1. Test every maximal pattern, exactly as the maximal mode does.
//   2. A maximal pattern that IS significant is a finding in its own right -- stop, and do
//      not test its sub-patterns.
//   3. A maximal pattern that is NOT significant is expanded into its immediate (k-1)
//      sub-patterns, which are then tested.
//   4. Descent continues only along branches that stay non-significant. A branch ends when
//      it becomes significant or reaches length 1.
//   5. A sub-pattern reachable from several parents is tested once.
//   6. Multiple-testing correction is applied over every pattern actually tested.
//   7. A child whose support equals its parent's marks the same neighborhoods as the
//      parent, so it carries no extra information and is not reported.

const NESTED_SUPPORT_TOL = 1e-9;
const NESTED_MAX_TESTED_PATTERNS = 5000;
const NESTED_MAX_EXPANSION_FACTOR = 50.0;

type TestedMotif = { motifs: string[], pValue: number, adjPValue?: number, ifSignificant?: boolean };

// Canonical motif key: the sorted motif.
const canonicalKey = (motif: string[]) => JSON.stringify(motif.toSorted());

type NestedMotifDescentParams<T> = {
  // Runs the hypergeometric test plus FDR correction over exactly the given motifs.
  // The per-motif statistics must not depend on which other motifs were in the same call.
  testMotifs: (motifs: string[][]) => Promise<T[]>,
  // Every frequent pattern (ifMax=false), used to find the maximal patterns and,
  // via their supports, the redundant sub-patterns (rule 7).
  frequentPatterns: FrequentPattern[],
  // A child is dropped when |support(child) - support(parent)| <= supportTol.
  supportTol?: number,
  // Guard rails: descent stops early rather than letting the tested set explode.
  maxTestedPatterns?: number,
  maxExpansionFactor?: number,
};

// Run the nested descent and return the corrected results over everything tested.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/9250b61/SpatialQuery/spatial_utils.py#L161
export async function nestedMotifDescent<T extends TestedMotif>(params: NestedMotifDescentParams<T>): Promise<T[]> {
  const {
    testMotifs,
    frequentPatterns,
    supportTol = NESTED_SUPPORT_TOL,
    maxTestedPatterns = NESTED_MAX_TESTED_PATTERNS,
    maxExpansionFactor = NESTED_MAX_EXPANSION_FACTOR,
  } = params;
  if (frequentPatterns.length === 0) {
    return [];
  }
  const supportOf = new Map(frequentPatterns.map(p => [canonicalKey(p.itemsets), p.support]));
  const maximal = findMaximalPatterns(frequentPatterns).map(p => p.itemsets.toSorted());

  // Level 0 is the maximal patterns; gating them decides where the descent starts.
  const gateRes = maximal.length > 0 ? await testMotifs(maximal) : [];
  if (gateRes.length === 0) {
    return [];
  }

  const significance = (results: T[]) => new Map(results.map(r => [canonicalKey(r.motifs), Boolean(r.ifSignificant)]));
  const gateSig = significance(gateRes);

  // `tested` records every pattern the descent touches, significant or not: correction
  // in rule 6 must cover them all. Insertion order keeps the final results reproducible.
  const tested = new Set(maximal.map(canonicalKey));
  const parentOf = new Map<string, string>();
  const levelTables: T[][] = [gateRes];
  let frontier = maximal.filter(m => !gateSig.get(canonicalKey(m)));

  while (frontier.length > 0) {
    const children: string[][] = [];
    for (const parent of frontier) {
      if (parent.length <= 1) continue;
      // Immediate sub-patterns, in the order of itertools.combinations(parent, len(parent) - 1).
      for (let drop = parent.length - 1; drop >= 0; drop--) {
        const child = parent.filter((_, n) => n !== drop);
        const childKey = canonicalKey(child);
        if (tested.has(childKey)) {
          // Reachable from an earlier parent already: test it once, and keep the
          // shallowest parent, which breadth-first order guarantees is this one.
          continue;
        }
        tested.add(childKey);
        parentOf.set(childKey, canonicalKey(parent));
        children.push(child);
      }
    }
    if (children.length === 0) break;

    if (tested.size > maxTestedPatterns || tested.size > maxExpansionFactor * Math.max(maximal.length, 1)) {
      console.warn(
        `Nested descent stopped early at ${tested.size} tested patterns `
        + `(limits: maxTestedPatterns=${maxTestedPatterns}, `
        + `maxExpansionFactor=${maxExpansionFactor} x ${maximal.length} maximal patterns). `
        + `Reported results cover the patterns reached so far.`,
      );
      break;
    }

    const levelRes = await testMotifs(children);
    levelTables.push(levelRes);
    const levelSig = significance(levelRes);
    frontier = children.filter(c => !levelSig.get(canonicalKey(c)));
  }

  // Every field except the FDR ones depends only on the motif and the data, so the
  // results computed level by level are reused; only the correction is redone over the union.
  const seen = new Set<string>();
  const combined = levelTables.flat().filter((r) => {
    const key = canonicalKey(r.motifs);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  // Rule 7: a child marking the same neighborhoods as its parent adds nothing. It stays
  // in the correction -- it was tested -- but is not reported.
  const redundant = new Set<string>();
  for (const [child, parent] of parentOf) {
    const supportChild = supportOf.get(child);
    const supportParent = supportOf.get(parent);
    if (supportChild !== undefined && supportParent !== undefined && Math.abs(supportChild - supportParent) <= supportTol) {
      redundant.add(child);
    }
  }

  return applyFdrCorrection(combined).filter(r => !redundant.has(canonicalKey(r.motifs)));
}
