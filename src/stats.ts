// Statistical helpers matching the scipy / statsmodels functions used by SpatialQuery,
// built on @stdlib/stats where it provides the functionality.
import padjust from '@stdlib/stats/padjust/lib/index.js';
import ranks from '@stdlib/stats/ranks/lib/index.js';
import stridedMean from '@stdlib/stats/strided/mean/lib/index.js';
import mediansorted from '@stdlib/stats/strided/mediansorted/lib/index.js';
import hypergeometricCdf from '@stdlib/stats/base/dists/hypergeometric/cdf/lib/index.js';
import hypergeometricMean from '@stdlib/stats/base/dists/hypergeometric/mean/lib/index.js';
import normalCdf from '@stdlib/stats/base/dists/normal/cdf/lib/index.js';

// Python's built-in round() uses round-half-to-even ("banker's rounding"),
// whereas Math.round rounds half up.
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
}

// Hypergeometric distribution, using the same parameterization as scipy.stats.hypergeom:
// M = total number of objects,
// n = total number of "success" objects,
// N = number of objects drawn without replacement.
// (In @stdlib, these are N, K, and n, respectively.)
// Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
export function hypergeom(M: number, n: number, N: number) {
  return {
    mean(): number {
      return hypergeometricMean(M, n, N);
    },
    // Survival function: P(X > k).
    // Computed as the CDF of the number of failures drawn, N - X ~ Hypergeometric(M, M - n, N):
    // P(X > k) = P(N - X <= N - k - 1). Unlike 1 - CDF, this retains precision for very small p-values.
    sf(k: number): number {
      return hypergeometricCdf(N - Math.floor(k) - 1, M, M - n, N);
    },
  };
}

// Survival function of the standard normal distribution, P(Z > z).
export function normSf(z: number): number {
  return normalCdf(-z, 0, 1);
}

export function mean(values: number[]): number {
  return stridedMean(values.length, values, 1);
}

export function median(values: number[]): number {
  const sorted = values.toSorted((a, b) => a - b);
  return mediansorted(sorted.length, sorted, 1);
}

// Ranks of the values (1-based), with ties assigned their average rank
// (equivalent to scipy.stats.rankdata(method='average') and pandas rank()).
export function rankAverage(values: number[]): number[] {
  return ranks(values, { method: 'average' });
}

// Number of arrangements yielding each value of the Mann-Whitney U statistic (0..n1*n2),
// for sample sizes n1 and n2. These are the coefficients of the Gaussian binomial
// coefficient [n1+n2 choose n1]_q = prod_{i=1..n1} (1 - q^(n2+i)) / (1 - q^i).
function mwuFrequencies(n1: number, n2: number): Float64Array {
  const maxU = n1 * n2;
  const freqs = new Float64Array(maxU + 1);
  freqs[0] = 1;
  for (let i = 1; i <= n1; i++) {
    // Multiply by 1 / (1 - q^i) = 1 + q^i + q^2i + ...
    for (let u = i; u <= maxU; u++) freqs[u] += freqs[u - i];
    // Multiply by (1 - q^(n2+i)).
    for (let u = maxU; u >= n2 + i; u--) freqs[u] -= freqs[u - n2 - i];
  }
  return freqs;
}

// Two-sided Mann-Whitney U test (not available in @stdlib/stats).
// Equivalent to scipy.stats.mannwhitneyu(x, y, alternative='two-sided', method='auto'):
// the exact null distribution is used when either sample has at most 8 values and there are no ties,
// otherwise the normal approximation with tie and continuity corrections.
// Reference: https://github.com/scipy/scipy/blob/main/scipy/stats/_mannwhitneyu.py
export function mannWhitneyU(x: number[], y: number[]): { statistic: number, pValue: number } {
  const n1 = x.length;
  const n2 = y.length;
  const xy = [...x, ...y];
  const r = rankAverage(xy);
  let r1 = 0;
  for (let i = 0; i < n1; i++) r1 += r[i];
  const u1 = r1 - n1 * (n1 + 1) / 2;
  const u2 = n1 * n2 - u1;

  // Number of values in each group of tied values.
  const valueCounts = new Map<number, number>();
  for (const v of xy) valueCounts.set(v, (valueCounts.get(v) ?? 0) + 1);
  const tieCounts = Array.from(valueCounts.values());
  const hasTies = tieCounts.some(t => t > 1);

  let pValue: number;
  if ((n1 <= 8 || n2 <= 8) && !hasTies) {
    // Exact: p = 2 * P(U >= max(U1, U2)) = 2 * P(U <= min(U1, U2)) by symmetry.
    const uMin = Math.round(Math.min(u1, u2));
    const freqs = mwuFrequencies(Math.min(n1, n2), Math.max(n1, n2));
    let count = 0;
    let total = 0;
    freqs.forEach((f, u) => {
      total += f;
      if (u <= uMin) count += f;
    });
    pValue = 2 * count / total;
  } else {
    const n = n1 + n2;
    const mu = n1 * n2 / 2;
    const tieTerm = tieCounts.reduce((acc, t) => acc + t ** 3 - t, 0);
    const s = Math.sqrt(n1 * n2 / 12 * ((n + 1) - tieTerm / (n * (n - 1))));
    // Continuity correction, in the direction that increases the p-value.
    const numerator = u1 - mu - 0.5 * Math.sign(u1 - mu);
    const z = numerator / s;
    pValue = 2 * normSf(Math.abs(z));
  }
  // NaN (e.g., when all values are tied) is propagated, as in scipy.
  return { statistic: u1, pValue: Math.min(Math.max(pValue, 0), 1) };
}

// Benjamini-Hochberg FDR correction.
// Equivalent to statsmodels.stats.multitest.fdrcorrection(pvals, alpha, method='poscorr').
// Returns the rejection decisions and adjusted p-values, in the original order.
export function fdrCorrection(pvals: number[], alpha = 0.05) {
  const adjusted = padjust(pvals, 'bh');
  const rejected = adjusted.map(p => p <= alpha);
  return { rejected, adjusted };
}

type FdrCorrectable = { pValue: number, adjPValue?: number, ifSignificant?: boolean };

// Determine significance across all tested motifs, applying FDR correction
// when more than one motif was tested.
// Reference: https://github.com/ShaokunAn/Spatial-Query/blob/0570c54bd6e046466e05c6b800d6701a7ab15c4a/SpatialQuery/spatial_utils.py#L134
export function applyFdrCorrection<T extends FdrCorrectable>(results: T[], alpha = 0.05): T[] {
  if (results.length === 0) {
    return [];
  }
  if (results.length === 1) {
    // Single-motif case: significance comes straight from the raw p-value, with no adjusted p-value.
    const { adjPValue: _, ...result } = results[0];
    return [{ ...result, ifSignificant: result.pValue < alpha } as T];
  }
  const { rejected, adjusted } = fdrCorrection(results.map(r => r.pValue), alpha);
  return results
    .map((r, i) => ({ ...r, adjPValue: adjusted[i], ifSignificant: rejected[i] }))
    .toSorted((a, b) => a.adjPValue - b.adjPValue);
}
