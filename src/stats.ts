// Statistical helpers ported from scipy / statsmodels,
// as used by SpatialQuery for motif enrichment.

// Cache exact log-factorials for small values; use Stirling's series beyond.
const LOG_FACTORIAL_CACHE_SIZE = 256;
const logFactorialCache = new Float64Array(LOG_FACTORIAL_CACHE_SIZE);
for (let i = 1; i < LOG_FACTORIAL_CACHE_SIZE; i++) {
  logFactorialCache[i] = logFactorialCache[i - 1] + Math.log(i);
}

const HALF_LOG_2PI = 0.5 * Math.log(2 * Math.PI);

// Natural log of k! for non-negative integer k.
export function logFactorial(k: number): number {
  if (k < LOG_FACTORIAL_CACHE_SIZE) {
    return logFactorialCache[k];
  }
  // Stirling's series for ln(Gamma(x)) with x = k + 1.
  const x = k + 1;
  const x2 = x * x;
  return (x - 0.5) * Math.log(x) - x + HALF_LOG_2PI
    + 1 / (12 * x)
    - 1 / (360 * x * x2)
    + 1 / (1260 * x * x2 * x2)
    - 1 / (1680 * x * x2 * x2 * x2);
}

// Natural log of the binomial coefficient (n choose k).
function logChoose(n: number, k: number): number {
  return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
}

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
// Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
export function hypergeom(M: number, n: number, N: number) {
  const lower = Math.max(0, N - (M - n));
  const upper = Math.min(n, N);
  const logDenom = logChoose(M, N);

  function logPmf(k: number): number {
    if (k < lower || k > upper) return -Infinity;
    return logChoose(n, k) + logChoose(M - n, N - k) - logDenom;
  }

  return {
    mean(): number {
      return M === 0 ? NaN : N * n / M;
    },
    pmf(k: number): number {
      return Math.exp(logPmf(k));
    },
    // Survival function: P(X > k).
    sf(k: number): number {
      const start = Math.max(Math.floor(k) + 1, lower);
      if (start > upper) return 0;
      // Sum the pmf over the upper tail in log-space (log-sum-exp)
      // to retain precision for very small p-values.
      const logTerms = [];
      let maxLogTerm = -Infinity;
      for (let j = start; j <= upper; j++) {
        const t = logPmf(j);
        logTerms.push(t);
        if (t > maxLogTerm) maxLogTerm = t;
      }
      if (maxLogTerm === -Infinity) return 0;
      let total = 0;
      for (const t of logTerms) {
        total += Math.exp(t - maxLogTerm);
      }
      return Math.min(1, Math.exp(maxLogTerm + Math.log(total)));
    },
  };
}

const SQRT_PI = Math.sqrt(Math.PI);

// Complementary error function, to near double precision.
function erfc(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x < 0) return 2 - erfc(-x);
  if (x < 3) {
    // Series erf(x) = 2/sqrt(pi) * exp(-x^2) * sum_n x (2x^2)^n / (1*3*...*(2n+1)),
    // whose terms are all positive (no cancellation).
    let term = x;
    let sum = x;
    for (let n = 1; term > sum * 1e-17; n++) {
      term *= 2 * x * x / (2 * n + 1);
      sum += term;
    }
    return 1 - 2 / SQRT_PI * Math.exp(-x * x) * sum;
  }
  if (x > 27) return 0;
  // Continued fraction erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...)))),
  // evaluated from the tail.
  let f = x;
  for (let n = 60; n >= 1; n--) {
    f = x + (n / 2) / f;
  }
  return Math.exp(-x * x) / SQRT_PI / f;
}

// Survival function of the standard normal distribution, P(Z > z).
export function normSf(z: number): number {
  return 0.5 * erfc(z / Math.SQRT2);
}

// Ranks of the values (1-based), with ties assigned their average rank.
// Equivalent to scipy.stats.rankdata(method='average') and pandas rank().
export function rankAverage(values: number[]): { ranks: number[], tieCounts: number[] } {
  const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const ranks = new Array<number>(values.length);
  const tieCounts: number[] = [];
  for (let start = 0; start < order.length;) {
    let end = start + 1;
    while (end < order.length && values[order[end]] === values[order[start]]) end++;
    // Positions start..end-1 (0-based) share the average of ranks start+1..end.
    const avgRank = (start + 1 + end) / 2;
    for (let p = start; p < end; p++) ranks[order[p]] = avgRank;
    tieCounts.push(end - start);
    start = end;
  }
  return { ranks, tieCounts };
}

export function median(values: number[]): number {
  const sorted = values.toSorted((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Number of arrangements yielding each value of the Mann-Whitney U statistic (0..maxU),
// for sample sizes n1 and n2. These are the coefficients of the Gaussian binomial
// coefficient [n1+n2 choose n1]_q = prod_{i=1..n1} (1 - q^(n2+i)) / (1 - q^i).
function mwuFrequencies(n1: number, n2: number, maxU: number): Float64Array {
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

// Binomial coefficient (n choose k), as a float.
function choose(n: number, k: number): number {
  return Math.exp(logFactorial(n) - logFactorial(k) - logFactorial(n - k));
}

// Two-sided Mann-Whitney U test.
// Equivalent to scipy.stats.mannwhitneyu(x, y, alternative='two-sided', method='auto'):
// the exact null distribution is used when both samples have at most 8 values and there are no ties,
// otherwise the normal approximation with tie and continuity corrections.
// Reference: https://github.com/scipy/scipy/blob/main/scipy/stats/_mannwhitneyu.py
export function mannWhitneyU(x: number[], y: number[]): { statistic: number, pValue: number } {
  const n1 = x.length;
  const n2 = y.length;
  const { ranks, tieCounts } = rankAverage([...x, ...y]);
  let r1 = 0;
  for (let i = 0; i < n1; i++) r1 += ranks[i];
  const u1 = r1 - n1 * (n1 + 1) / 2;
  const u2 = n1 * n2 - u1;
  const hasTies = tieCounts.some(t => t > 1);

  let pValue: number;
  if ((n1 <= 8 || n2 <= 8) && !hasTies) {
    // Exact: p = 2 * P(U >= max(U1, U2)) = 2 * P(U <= min(U1, U2)) by symmetry.
    const uMin = Math.round(Math.min(u1, u2));
    const freqs = mwuFrequencies(Math.min(n1, n2), Math.max(n1, n2), uMin);
    let count = 0;
    for (let u = 0; u <= uMin; u++) count += freqs[u];
    pValue = 2 * count / choose(n1 + n2, n1);
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
  const m = pvals.length;
  const order = pvals.map((_, i) => i).sort((a, b) => pvals[a] - pvals[b]);
  const adjusted = new Array<number>(m);
  let runningMin = 1;
  // Step-up: enforce monotonicity from the largest p-value downward.
  for (let rank = m; rank >= 1; rank--) {
    const i = order[rank - 1];
    runningMin = Math.min(runningMin, pvals[i] * m / rank);
    adjusted[i] = runningMin;
  }
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
