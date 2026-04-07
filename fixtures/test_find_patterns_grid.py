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
from os.path import join

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

    # We re-write using a newer version of the anndata package.
    # Certain things in the original secondary_analysis.h5ad file,
    # like categorical obs columns, cannot be read by
    # hdf5-as-virtual-zarr.js + anndata.js (at least currently).
    adata.write_h5ad("secondary_analysis.new.h5ad")

    label_key = "predicted_label"
    spatial_key = "X_spatial"
    feature_name = "hugo_symbol"
    normalize_spatial_coord = False

    sp = spatial_query(
        adata=adata,
        dataset="test",
        spatial_key=spatial_key,
        label_key=label_key,
        feature_name=feature_name,
        leaf_size=10,
        build_gene_index=False,
        if_lognorm=True,
        if_normalize_spatial_coord=normalize_spatial_coord,
    )

    cell_i_to_cell_id = dict(zip(range(adata.obs.shape[0]), adata.obs.index.tolist()))
    cell_id_to_cell_type = dict(zip(adata.obs.index.tolist(), adata.obs[label_key].tolist()))

    max_dist = 150
    min_support = 0.25

    fp_tree, grid_pos = sp.find_patterns_grid(
        max_dist=max_dist,
        min_support=min_support,
        if_display=False,
        figsize=(9, 6),
        return_cellID=True,
        return_grid=True,
    )

    #print(fp_tree)
    #print(grid_pos)
    #print(json.dumps(grid_pos, cls=NumpyEncoder))

    out_path = join("secondary_analysis.find_patterns_grid.json")

    with open(out_path, "w") as f:
        json.dump({
            "params": dict(
                spatial_key=spatial_key,
                label_key=label_key,
                leaf_size=10,
                normalize_spatial_coord=normalize_spatial_coord,
                max_dist=max_dist,
                min_support=min_support,
            ),
            "fp_tree": fp_tree.to_dict("records"),
            "grid_pos": grid_pos,
        }, f, cls=NumpyEncoder, indent=2)

    # print(fp_tree.to_dict("records"))
    #print(json.dumps(fp_tree.to_dict("records"), cls=NumpyEncoder))

    # TODO: also write the inputs to json,
    # to easily use in Vitest unit tests as inputs to the JS functions.
