// TODO: use vitest to compare the outputs of the Python find_patterns_grid to the JS findPatternsGrid.
import { describe, it, expect } from "vitest";
import { fileURLToPath } from "url";
import { resolve, dirname, join } from "path";

import * as zarr from 'zarrita';
import FileSystemStore from '@zarrita/storage/fs';
import { HdfStore } from "hdf5-as-virtual-zarr";
import { readZarr, get, readElem } from 'anndata.js';
import { assert } from './assert.js';


const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const fixturesDir = resolve(__dirname, "..", "fixtures");


async function getData() {

}


describe("Compare Python outputs to JS outputs", () => {
  // Reference: https://github.com/keller-mark/hdf5-as-virtual-zarr.js/blob/main/test/read-adata.test.ts
  it("findPatternsGrid", async () => {
    const internalStore = new FileSystemStore(join(fixturesDir, "secondary_analysis.new.h5ad"));
    const store = await HdfStore.fromStore(internalStore);

    // We use a patched anndata.js due to
    // https://github.com/ilan-gold/anndata.js/issues/52
    const adata = await readZarr(store);

    const obsIndexArr = await adata.obsNames();
    const obsIndex = await get(obsIndexArr, [null]);

    expect(obsIndex.shape).toEqual([4058]);

    const col = await zarr.open(zarr.root(store).resolve("/obs/predicted_label"), { kind: "group" });
    console.log(col.attrs);

    const predictedLabel = await adata.obs.get("predicted_label");
    assert("codes" in predictedLabel, "Was not categorical");

    const labelCodes = await get(predictedLabel.codes, [null]);
    const labelCats = await get(predictedLabel.categories, [null]);

    const labels = Array.from(labelCodes.data).map(code => labelCats.data[code]);


    const spatialCoordsArr = await adata.obsm.get("X_spatial");
    assert(!("codes" in spatialCoordsArr), "Was unexpectedly categorical");
    assert(!("indptr" in spatialCoordsArr), "Was unexpectedly sparse");
    assert(!("axisRoot" in spatialCoordsArr), "Was unexpectedly axis_arrays");

    const spatialX = await get(spatialCoordsArr, [null, 0]);
    const spatialY = await get(spatialCoordsArr, [null, 1]);

    console.log(spatialX.data);





  });
});
