import { describe, it, expect } from "vitest";
import { normalizeSpatialCoords, buildContext } from "../src/index.js";

// Deterministic point set, also used to generate the expected value with
// spatial_utils._auto_normalize_spatial_coords (scipy KDTree, k=2):
//   i = np.arange(500)
//   pts = np.stack([(i*37%101)+np.sin(i), (i*53%97)+np.cos(i)], axis=1)
const N = 500;
const xCoords = Float64Array.from({ length: N }, (_, i) => (i * 37 % 101) + Math.sin(i));
const yCoords = Float64Array.from({ length: N }, (_, i) => (i * 53 % 97) + Math.cos(i));
const EXPECTED_SCALE_FACTOR = 0.5429866729645025;

describe("normalizeSpatialCoords", () => {
  it("matches the Python scale factor", () => {
    const result = normalizeSpatialCoords(xCoords, yCoords);
    expect(result.scaleFactor).toBeCloseTo(EXPECTED_SCALE_FACTOR, 12);
    expect(result.xCoords[3]).toBeCloseTo(xCoords[3] * EXPECTED_SCALE_FACTOR, 12);
    expect(result.yCoords[3]).toBeCloseTo(yCoords[3] * EXPECTED_SCALE_FACTOR, 12);
  });
});

describe("buildContext", () => {
  const obsIndex = Array.from({ length: N }, (_, i) => `cell${i}`);
  const params = { obsIndex, xCoords, yCoords, obsIdToObsSetName: () => "A" };

  it("uses normalized coordinates by default", () => {
    const ctx = buildContext(params);
    const [xMin, xMax] = ctx.xExtent;
    expect(xMin).toBeCloseTo(Math.min(...xCoords) * EXPECTED_SCALE_FACTOR, 12);
    expect(xMax).toBeCloseTo(Math.max(...xCoords) * EXPECTED_SCALE_FACTOR, 12);
    expect(ctx.xCoords[10]).toBeCloseTo(xCoords[10] * EXPECTED_SCALE_FACTOR, 12);
  });

  it("keeps original coordinates when normalizeSpatialCoord is false", () => {
    const ctx = buildContext({ ...params, normalizeSpatialCoord: false });
    expect(ctx.xExtent).toEqual([Math.min(...xCoords), Math.max(...xCoords)]);
    expect(ctx.xCoords).toBe(xCoords);
  });
});
