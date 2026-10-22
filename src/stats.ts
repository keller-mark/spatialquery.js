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
