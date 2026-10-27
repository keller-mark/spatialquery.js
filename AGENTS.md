This is a TypeScript port of the Python `Spatial-Query` method.

The original python implementation is contained at `./Spatial-Query` as a git submodule, for reference during porting.
The primary file to reference is `./Spatial-Query/SpatialQuery/spatial_query.py`.
We are porting the following functions:
- find_patterns_grid
- find_patterns_rand
- motif_enrichment_knn
- motif_enrichment_dist
- motif_composition_knn
- motif_composition_dist

As well as their multi-FOV variants (`./Spatial-Query/SpatialQuery/spatial_query_multiple_fov.py`, class `spatial_query_multi`), including the pattern-level differential analysis between datasets (`differential_analysis_knn/dist`, `./Spatial-Query/SpatialQuery/spatial_differential_pattern.py`).

Out of scope: gene expression analyses (`de_genes`, gene-gene covarying), and plotting (Vitessce handles visualization).

## Layout

- `src/index.ts`: public exports.
- `src/single-fov.ts`: context (`buildContext`), neighborhood queries, and the single-FOV functions (`spatial_query`).
- `src/multi-fov.ts`: multi-FOV context (`buildMultiContext`) and functions (`spatial_query_multi`), including differential analysis.
- `src/nested.ts`: nested motif enrichment (`nestedMotifDescent`).
- `src/fpgrowth.ts`: frequent pattern mining (`node-fpgrowth`, made to match mlxtend `fpgrowth` + `find_maximal_patterns`).
- `src/stats.ts`: scipy/statsmodels-compatible wrappers around `@stdlib/stats` (`hypergeom`, `fdrCorrection` via `padjust(..., 'bh')`, `normSf`, `rankAverage`, `mean`, `median`), plus `mannWhitneyU` (scipy `mannwhitneyu`, `method='auto'`; not in stdlib), `applyFdrCorrection`, and `roundHalfEven` (Python `round`).
- `src/stdlib.d.ts`: type declarations for the `@stdlib/stats` imports (see Conventions).
- `fixtures/*.py`: UV scripts that generate the Python reference outputs (JSON), compared against in `test/python-parity.test.ts` and `test/multi-fov-parity.test.ts` (shared helpers in `test/fixture-data.ts`). The multi-FOV fixtures split the HuBMAP dataset into 16 FOVs across two datasets.

## Conventions

- Spatial index: Flatbush. Radius queries via `neighbors(x, y, Infinity, maxDist)` (inclusive, like `query_ball_point`); k-NN via `neighbors(x, y, k, Infinity, j => j !== i)` (like `query(k=k+1)` minus the cell itself).
- Results use camelCase field names; cell IDs are returned as obs ID strings (`neighborIds`, `centerIds`), not integer indices. Multi-FOV cell IDs are keyed by dataset ID (`Record<string, string[]>`), like Python's `{dataset: [ids]}` dicts.
- Multi-FOV datasets: each FOV has an explicit, unique `datasetId` and a `datasetClass` (e.g., condition) shared by related FOVs. Python instead takes one (possibly repeated) name per FOV and derives unique FOV names internally (`'a_b'` -> `'a-b_0'`, `'a-b_1'`, ...), recovering the class with `name.split('_')[0]`. Python's `dataset` filter corresponds to `datasetClass`, `dataset_i` to `datasetId`, and the differential analysis `datasets` to `datasetClasses`.
- Multi-FOV functions take a `SqMultiContext` and are prefixed with `multi` when a single-FOV function of the same name exists (`multiMotifEnrichmentKnn`, ...).
- Statistics: use `@stdlib/stats` where it provides the function rather than implementing it. Import the entry file of each subpath (e.g. `@stdlib/stats/padjust/lib/index.js`), since the package has no `exports` map and Node's ESM loader rejects directory imports like `@stdlib/stats/padjust`; declare each such import in `src/stdlib.d.ts`, re-using the subpath's types.
- `findPatternsRand` uses `Math.random` (no numpy-compatible seeded RNG); its tests must be robust to randomness.

## Known deviations from Python

- `find_patterns_grid` / `find_patterns_rand`: Python's `build_fptree_dist` (with `cinds=None`) also drops the cell whose index equals the query point's index. Not reproduced.
- `motif_enrichment_*` with `return_cellID=True`: Python silently drops motifs with no center cells; JS keeps them with empty cell IDs.
- Cells at identical coordinates: k-NN may exclude a different cell than scipy's "drop the first result".
- Patterns with tied support / adjusted p-values may be ordered differently.
- Multi-FOV: invalid `datasetClass` values throw in every method (Python only validates dataset names in `find_fp_*` and the differential analysis).
- Multi-FOV `find_fp_knn` with `max_dist <= 0`: Python adds empty transactions; JS adds none (as in single-FOV).
- Differential analysis: duplicate motifs are tested once (Python appends their supports twice to the same list). Results also include `differentialTestsKnn/Dist`, which return every tested motif (not just the significant ones).

## Plan

1. [x] `buildContext` with `normalizeSpatialCoord`.
2. [x] Hypergeometric test + FDR correction (`computeMotifEnrichment`, `applyFdrCorrection`).
3. [x] `findPatternsGrid` (+ `buildGridPoints`), `findPatternsRand`.
4. [x] `findFpKnn`, `findFpDist` (with `ifMax`), `findMaximalPatterns`.
5. [x] `motifEnrichmentKnn`, `motifEnrichmentDist` (`mode='maximal'`, `returnCellIds`).
6. [x] `mode='nested'` for motif enrichment (`nestedMotifDescent`).
7. [x] `motifCompositionKnn`, `motifCompositionDist`.
8. [x] Multi-FOV: `buildMultiContext`, `multiFindFpKnn/Dist`, `findFpKnnFov/DistFov`, `multiMotifEnrichmentKnn/Dist` (incl. nested), `multiMotifCompositionKnn/Dist`, `differentialAnalysisKnn/Dist`.
9. [x] Python reference fixtures + parity tests for 6–8.

Possible next steps:
- Performance on large datasets (e.g. a Web Worker, or caching neighborhoods across calls with the same `k` / `maxDist`).
- Remove the unused `internmap` dependency.
