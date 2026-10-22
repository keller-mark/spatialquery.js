Download test data fixtures to enable running the JS unit tests, as well as to enable comparisons between the Python and JS implementations.


### Download hubmap example files

Download the `secondary_analysis.h5ad` file for sample `HBM838.LDFP.578` from the HuBMAP Portal at https://portal.hubmapconsortium.org/browse/dataset/5bf1e7b295343c4537206beda25aa4ca

- https://assets.hubmapconsortium.org/0a21f3fa27109790483f2a0729be53de/secondary_analysis.h5ad


### Generate python outputs

```sh
CC=/usr/bin/clang CXX=/usr/bin/clang++ uv run test_find_patterns_grid.py
CC=/usr/bin/clang CXX=/usr/bin/clang++ uv run generate_motif_fixtures.py
CC=/usr/bin/clang CXX=/usr/bin/clang++ uv run generate_multi_fov_fixtures.py
```
