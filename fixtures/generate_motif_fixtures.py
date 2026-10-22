#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "anndata>=0.7.8",
#   "pybind11",
#   "SpatialQuery @ git+https://github.com/ShaokunAn/Spatial-Query@9250b61"
# ]
# ///
#
# Generates Python reference outputs for find_patterns_rand, find_fp_knn, find_fp_dist,
# motif_enrichment_knn and motif_enrichment_dist, for comparison with the JS port.
#
# When installing with `uv`, may need --no-build-isolation-package SpatialQuery

from anndata import read_h5ad
from SpatialQuery import spatial_query
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def records(df):
    return df.to_dict("records")


if __name__ == "__main__":
    adata = read_h5ad("secondary_analysis.h5ad")

    label_key = "predicted_label"
    spatial_key = "X_spatial"
    normalize_spatial_coord = False

    sp = spatial_query(
        adata=adata,
        dataset="test",
        spatial_key=spatial_key,
        label_key=label_key,
        feature_name="hugo_symbol",
        leaf_size=10,
        build_gene_index=False,
        if_lognorm=True,
        if_normalize_spatial_coord=normalize_spatial_coord,
    )

    # Use the most abundant cell type as the center cell type.
    ct = adata.obs[label_key].value_counts().index[0]

    max_dist = 150
    k = 30
    min_support = 0.25
    min_size = 0
    n_points = 1000
    seed = 2023

    rand_fp = sp.find_patterns_rand(
        max_dist=max_dist, n_points=n_points, min_support=min_support, min_size=min_size,
        if_display=False, return_cellID=True, seed=seed,
    )
    fp_knn = sp.find_fp_knn(ct=ct, k=k, min_support=min_support, max_dist=max_dist)
    fp_dist = sp.find_fp_dist(ct=ct, max_dist=max_dist, min_size=min_size, min_support=min_support)

    enrich_knn = sp.motif_enrichment_knn(
        ct=ct, k=k, min_support=min_support, max_dist=max_dist, return_cellID=True,
    )
    enrich_dist = sp.motif_enrichment_dist(
        ct=ct, max_dist=max_dist, min_size=min_size, min_support=min_support, return_cellID=True,
    )
    # Also test user-specified motifs, including one that contains the center cell type.
    other_ct = adata.obs[label_key].value_counts().index[1]
    specified_motifs = [[other_ct], [ct, other_ct]]
    enrich_knn_specified = sp.motif_enrichment_knn(
        ct=ct, motifs=specified_motifs, k=k, max_dist=max_dist,
    )
    enrich_dist_specified = sp.motif_enrichment_dist(
        ct=ct, motifs=specified_motifs, max_dist=max_dist,
    )

    with open("secondary_analysis.motif_fixtures.json", "w") as f:
        json.dump({
            "params": dict(
                spatial_key=spatial_key,
                label_key=label_key,
                leaf_size=10,
                normalize_spatial_coord=normalize_spatial_coord,
                ct=ct,
                max_dist=max_dist,
                k=k,
                min_support=min_support,
                min_size=min_size,
                n_points=n_points,
                seed=seed,
                specified_motifs=specified_motifs,
            ),
            "find_patterns_rand": records(rand_fp),
            "find_fp_knn": records(fp_knn),
            "find_fp_dist": records(fp_dist),
            "motif_enrichment_knn": records(enrich_knn),
            "motif_enrichment_dist": records(enrich_dist),
            "motif_enrichment_knn_specified": records(enrich_knn_specified),
            "motif_enrichment_dist_specified": records(enrich_dist_specified),
        }, f, cls=NumpyEncoder, indent=2)
