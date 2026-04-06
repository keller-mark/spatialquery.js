#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "anndata>=0.7.8",
#   "pybind11",
#   "SpatialQuery @ git+https://github.com/ShaokunAn/Spatial-Query@9dde563"
# ]
# ///
#
# When installing with `uv`, may need --no-build-isolation-package SpatialQuery

from anndata import read_h5ad
from SpatialQuery import spatial_query
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    print("Hello world")

    adata = read_h5ad("secondary_analysis.h5ad")

    label_key = "predicted_label"
    spatial_key = "X_spatial"
    feature_name = "hugo_symbol"

    sp = spatial_query(
        adata=adata,
        dataset="test",
        spatial_key=spatial_key,
        label_key=label_key,
        feature_name=feature_name,
        leaf_size=10,
        build_gene_index=False,
        if_lognorm=True,
        if_normalize_spatial_coord=False,
    )

    cell_i_to_cell_id = dict(zip(range(adata.obs.shape[0]), adata.obs.index.tolist()))
    cell_id_to_cell_type = dict(zip(adata.obs.index.tolist(), adata.obs[label_key].tolist()))

    max_dist = 20
    min_support = 0.5

    fp_tree, grid_pos = sp.find_patterns_grid(
        max_dist=max_dist,
        min_support=min_support,
        if_display=False,
        figsize=(9, 6),
        return_cellID=True,
        return_grid=True,
    )

    print(fp_tree)

    # print(fp_tree.to_dict("records"))
    print(json.dumps(fp_tree.to_dict("records"), cls=NumpyEncoder))

    # TODO: also write the inputs to json,
    # to easily use in Vitest unit tests as inputs to the JS functions.
