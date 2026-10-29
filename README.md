# spatialquery.js
JavaScript implementation of [Spatial-Query](https://github.com/ShaokunAn/Spatial-Query).


## Citation

If you use SpatialQuery in your research, please cite the [preprint](https://doi.org/10.64898/2026.04.22.720136):


> Shaokun An, Mark Keller, Nils Gehlenborg, Martin Hemberg. **SpatialQuery: scalable discovery and molecular characterization of multicellular motifs from spatial omics data.** bioRxiv. doi: 10.64898/2026.04.22.720136


## Installation

```sh
npm install spatialquery.js
```

## Usage

### Single field of view

```js
import {
  buildContext,
  findPatternsGrid,
  motifEnrichmentDist,
  motifCompositionDist,
} from 'spatial-query';

// Cell type label of each cell.
const obsIdToLabel = new Map(obsIndex.map((obsId, i) => [obsId, labels[i]]));

const context = buildContext({
  obsIndex,          // string[] of cell IDs
  xCoords,           // Float32Array | Float64Array
  yCoords,           // Float32Array | Float64Array
  obsIdToObsSetName: obsId => obsIdToLabel.get(obsId),
  // By default, coordinates are normalized so the mean nearest-neighbor distance is 1.
  normalizeSpatialCoord: true,
});

// Frequent cell type patterns around grid points.
const patterns = await findPatternsGrid(context, { maxDist: 20, minSupport: 0.5 });
// => [{ itemsets: ['B cell', 'T cell'], support: 0.62 }, ...]

// Motifs enriched around a center cell type (k-NN version: motifEnrichmentKnn).
const enrichment = await motifEnrichmentDist(context, {
  ct: 'T cell',
  maxDist: 20,
  returnCellIds: true,  // adds centerIds / neighborIds (obs IDs) to each result
});
// => [{ center, motifs, nCenterMotif, nCenter, nMotif, expectation, pValue, adjPValue, ifSignificant, ... }]

// Composition of a motif in the neighborhoods of the center cells.
const composition = motifCompositionDist(context, {
  ct: 'T cell',
  motifs: enrichment[0].motifs,
  maxDist: 20,
});
```

Other functions: `findPatternsRand`, `findFpKnn`, `findFpDist`, `motifEnrichmentKnn`, `motifCompositionKnn`.
The enrichment functions also accept `motifs` (to test specific motifs) and `mode: 'nested'` (to also test sub-patterns of non-significant motifs).

### Multiple fields of view

```js
import {
  buildMultiContext,
  multiMotifEnrichmentKnn,
  differentialAnalysisKnn,
} from 'spatial-query';

// One entry per FOV, with a unique dataset ID and a class (e.g., condition) shared by related FOVs.
const multi = buildMultiContext([
  { datasetId: 'sample-1', datasetClass: 'control', obsIndex: obsIndex1, xCoords: x1, yCoords: y1, obsIdToObsSetName: toLabel1 },
  { datasetId: 'sample-2', datasetClass: 'control', obsIndex: obsIndex2, xCoords: x2, yCoords: y2, obsIdToObsSetName: toLabel2 },
  { datasetId: 'sample-3', datasetClass: 'disease', obsIndex: obsIndex3, xCoords: x3, yCoords: y3, obsIdToObsSetName: toLabel3 },
  // ...
]);

// Enrichment pooled across the FOVs of the selected classes (all FOVs by default).
// With returnCellIds: true, cell IDs are keyed by dataset ID.
const enrichment = await multiMotifEnrichmentKnn(multi, { ct: 'T cell', k: 30, maxDist: 20, datasetClass: 'disease' });

// Motifs whose support differs significantly between the FOVs of two classes.
const differential = await differentialAnalysisKnn(multi, {
  ct: 'T cell',
  datasetClasses: ['control', 'disease'],
});
// => { control: [{ itemsets, supportMean, adjPValue }, ...], disease: [...] }
```

Other functions: `multiFindFpKnn`, `multiFindFpDist`, `findFpKnnFov` / `findFpDistFov` (one FOV, by `datasetId`), `multiMotifEnrichmentDist`, `multiMotifCompositionKnn`, `multiMotifCompositionDist`, `differentialAnalysisDist`, and `differentialTestsKnn` / `differentialTestsDist` (all tested motifs, not only the significant ones).


## Development

### Setup

```sh
pnpm install
```

Download the test fixture files and run the Python fixture conversion/generation scripts:

```sh
cd fixtures
# The HuBMAP assets server rejects curl's default User-Agent.
curl -fL -A "Mozilla/5.0" -o secondary_analysis.h5ad https://assets.hubmapconsortium.org/0a21f3fa27109790483f2a0729be53de/secondary_analysis.h5ad
CC=/usr/bin/clang CXX=/usr/bin/clang++ uv run test_find_patterns_grid.py
CC=/usr/bin/clang CXX=/usr/bin/clang++ uv run generate_motif_fixtures.py
CC=/usr/bin/clang CXX=/usr/bin/clang++ uv run generate_multi_fov_fixtures.py
cd ..
```

### Testing

The tests compare the JS outputs to Python reference outputs. See [fixtures/README.md](fixtures/README.md) to download the test data and generate the reference outputs, then:

```sh
pnpm test
```

### Publish to NPM

```sh
npm login
pnpm publish --no-git-checks --access public
```

## Related work

Related to:
- Initial implementation work in [vitessce#2116](https://github.com/vitessce/vitessce/pull/2116/).
- Widget plugin in [vitessce-python](https://github.com/vitessce/vitessce-python/blob/main/src/vitessce/widget_plugins/spatial_query.py) ([example notebook](https://github.com/vitessce/vitessce-python/blob/main/docs/notebooks/widget_plugin_spatial-query.ipynb))
