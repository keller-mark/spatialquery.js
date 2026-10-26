// @stdlib/stats has no "exports" map, so Node's ESM loader cannot resolve subpath (directory)
// imports such as '@stdlib/stats/padjust'. We import the entry files directly instead,
// and re-use the type declarations of the corresponding subpaths.

declare module '@stdlib/stats/padjust/lib/index.js' {
  import padjust = require('@stdlib/stats/padjust');
  export default padjust;
}

declare module '@stdlib/stats/ranks/lib/index.js' {
  import ranks = require('@stdlib/stats/ranks');
  export default ranks;
}

declare module '@stdlib/stats/strided/mean/lib/index.js' {
  import mean = require('@stdlib/stats/strided/mean');
  export default mean;
}

declare module '@stdlib/stats/strided/mediansorted/lib/index.js' {
  import mediansorted = require('@stdlib/stats/strided/mediansorted');
  export default mediansorted;
}

declare module '@stdlib/stats/base/dists/hypergeometric/cdf/lib/index.js' {
  import cdf = require('@stdlib/stats/base/dists/hypergeometric/cdf');
  export default cdf;
}

declare module '@stdlib/stats/base/dists/hypergeometric/mean/lib/index.js' {
  import mean = require('@stdlib/stats/base/dists/hypergeometric/mean');
  export default mean;
}

declare module '@stdlib/stats/base/dists/normal/cdf/lib/index.js' {
  import cdf = require('@stdlib/stats/base/dists/normal/cdf');
  export default cdf;
}
