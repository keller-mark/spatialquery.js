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
# Generates Python reference outputs for the multi-FOV functions (spatial_query_multi),
# for comparison with the JS port.
#
# The single HuBMAP dataset is split into 16 FOVs: 8 bins along x (by rank) x 2 halves along y.
# The left 4 x-bins form the dataset "kidney_left" and the right 4 form "kidney_right"
# (renamed by SpatialQuery to "kidney-left_0", ..., "kidney-right_7").
#
# When installing with `uv`, may need --no-build-isolation-package SpatialQuery

from anndata import read_h5ad
from SpatialQuery import spatial_query_multi
import SpatialQuery.spatial_differential_pattern as sdp
from scipy import stats
from types import SimpleNamespace
import inspect
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

    # Assign each cell to a FOV.
    pos = np.asarray(adata.obsm[spatial_key])
    n = pos.shape[0]
    x_bin = (np.argsort(np.argsort(pos[:, 0])) * 8) // n
    y_bin = (pos[:, 1] > np.median(pos[:, 1])).astype(int)
    cell_fov = x_bin * 2 + y_bin
    fov_datasets = ["kidney_left" if fov // 2 < 4 else "kidney_right" for fov in range(16)]

    adatas = [adata[cell_fov == fov].copy() for fov in range(16)]
    spm = spatial_query_multi(
        adatas=adatas,
        datasets=fov_datasets,
        spatial_key=spatial_key,
        label_key=label_key,
        leaf_size=10,
        build_gene_index=False,
        feature_name="hugo_symbol",
        if_lognorm=True,
        if_normalize_spatial_coord=normalize_spatial_coord,
    )
    # Obs names of each FOV, to convert per-FOV cell indices to obs IDs.
    fov_obs_names = {ds: a.obs_names.tolist() for ds, a in zip(spm.datasets, adatas)}

    def ids_to_names(cell_ids):
        # {motif_str: {dataset: [cell indices]}} -> {motif_str: {dataset: [obs names]}}
        return {
            motif: {ds: sorted(fov_obs_names[ds][i] for i in ids) for ds, ids in by_ds.items()}
            for motif, by_ds in cell_ids.items()
        }

    ct = adata.obs[label_key].value_counts().index[0]
    other_ct = adata.obs[label_key].value_counts().index[1]
    specified_motifs = [[other_ct], [ct, other_ct]]

    max_dist = 150
    k = 30
    min_support = 0.25
    min_size = 0

    fp_knn = spm.find_fp_knn(ct=ct, k=k, min_support=min_support, max_dist=max_dist)
    fp_dist_left = spm.find_fp_dist(ct=ct, dataset="kidney-left", max_dist=max_dist,
                                    min_size=min_size, min_support=min_support)

    enrich_knn, enrich_knn_motif_ids, enrich_knn_center_ids = spm.motif_enrichment_knn(
        ct=ct, k=k, min_support=min_support, max_dist=max_dist, return_cellID=True,
    )
    enrich_dist, enrich_dist_motif_ids, enrich_dist_center_ids = spm.motif_enrichment_dist(
        ct=ct, max_dist=max_dist, min_size=min_size, min_support=min_support, return_cellID=True,
    )
    enrich_knn_specified_right = spm.motif_enrichment_knn(
        ct=ct, motifs=specified_motifs, dataset="kidney-right", k=k, max_dist=max_dist,
    )
    enrich_dist_nested = spm.motif_enrichment_dist(
        ct=ct, max_dist=max_dist, min_size=min_size, min_support=min_support, mode='nested',
    )

    composition_knn = spm.motif_composition_knn(ct=ct, motifs=specified_motifs, k=k, max_dist=max_dist)
    composition_dist_left = spm.motif_composition_dist(
        ct=ct, motifs=specified_motifs, dataset="kidney-left", max_dist=max_dist,
    )

    fp_knn_fov = spm.find_fp_knn_fov(ct=ct, dataset_i="kidney-left_0", k=k,
                                     min_support=min_support, max_dist=max_dist)
    fp_dist_fov = spm.find_fp_dist_fov(ct=ct, dataset_i="kidney-right_3", max_dist=max_dist,
                                       min_size=min_size, min_support=min_support)

    # The differential analysis only returns significant motifs, so we also record every
    # Mann-Whitney U test (per-FOV supports of each motif, and the p-value).
    recorded_tests = []

    def recording_mannwhitneyu(x, y, **kwargs):
        res = stats.mannwhitneyu(x, y, **kwargs)
        motif = inspect.currentframe().f_back.f_locals['motif_sorted']
        recorded_tests.append({"itemsets": list(motif), "support_0": x, "support_1": y,
                               "p_value": res.pvalue})
        return res

    sdp.stats = SimpleNamespace(mannwhitneyu=recording_mannwhitneyu)

    def run_differential(fn, **kwargs):
        recorded_tests.clear()
        diff = fn(**kwargs)
        return {
            "significant": {ds: records(df) for ds, df in diff.items()},
            "tests": list(recorded_tests),
        }

    datasets = ["kidney-left", "kidney-right"]
    diff_knn = run_differential(spm.differential_analysis_knn, ct=ct, datasets=datasets, k=k,
                                min_support=min_support, max_dist=max_dist)
    diff_dist = run_differential(spm.differential_analysis_dist, ct=ct, datasets=datasets,
                                 max_dist=max_dist, min_support=min_support, min_size=min_size)
    diff_dist_specified = run_differential(spm.differential_analysis_dist, ct=ct, datasets=datasets,
                                           motifs=specified_motifs, max_dist=max_dist)

    # Reference values for the Mann-Whitney U test, covering the exact and asymptotic methods.
    mwu_cases = [
        ([0.1, 0.4, 0.35, 0.8], [0.2, 0.05, 0.15]),  # exact
        ([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.97], [0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.65, 0.75]),  # exact
        ([0.1, 0.1, 0.4, 0.5], [0.2, 0.1, 0.0]),  # ties -> asymptotic
        (list(np.linspace(0, 1, 12)), list(np.linspace(0.3, 1.5, 10))),  # large -> asymptotic
        # Note: all-tied inputs give a NaN p-value, which cannot be stored in JSON.
    ]
    mwu_reference = [
        {"x": x, "y": y, "p_value": stats.mannwhitneyu(x, y, alternative='two-sided', method='auto').pvalue}
        for x, y in mwu_cases
    ]

    with open("secondary_analysis.multi_fov_fixtures.json", "w") as f:
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
                specified_motifs=specified_motifs,
                datasets=datasets,
            ),
            "cell_fov": cell_fov,
            "fov_datasets": fov_datasets,
            "renamed_datasets": spm.datasets,
            "find_fp_knn": records(fp_knn),
            "find_fp_dist_left": records(fp_dist_left),
            "motif_enrichment_knn": records(enrich_knn),
            "motif_enrichment_knn_motif_ids": ids_to_names(enrich_knn_motif_ids),
            "motif_enrichment_knn_center_ids": ids_to_names(enrich_knn_center_ids),
            "motif_enrichment_dist": records(enrich_dist),
            "motif_enrichment_dist_motif_ids": ids_to_names(enrich_dist_motif_ids),
            "motif_enrichment_dist_center_ids": ids_to_names(enrich_dist_center_ids),
            "motif_enrichment_knn_specified_right": records(enrich_knn_specified_right),
            "motif_enrichment_dist_nested": records(enrich_dist_nested),
            "motif_composition_knn": records(composition_knn),
            "motif_composition_dist_left": records(composition_dist_left),
            "find_fp_knn_fov": records(fp_knn_fov),
            "find_fp_dist_fov": records(fp_dist_fov),
            "differential_analysis_knn": diff_knn,
            "differential_analysis_dist": diff_dist,
            "differential_analysis_dist_specified": diff_dist_specified,
            "mwu_reference": mwu_reference,
        }, f, cls=NumpyEncoder, indent=2)
