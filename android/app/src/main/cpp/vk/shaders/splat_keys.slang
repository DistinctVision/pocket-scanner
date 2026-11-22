#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Splat2D> splats;

[[vk::binding(1, 0)]]
StructuredBuffer<int> offsets;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Key64> keys;

[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> ids;

struct KeyPush {
    int count;
    int tilesX;
    int tilesY;
    int width;
    int height;
    int pad0;
    int pad1;
    int pad2;
};

[[vk::push_constant]]
KeyPush pc;

void emit(int base, inout int cursor, int tile, float depth, uint id) {
    Key64 key;
    key.depth = asuint(depth);
    key.tile = uint(tile);
    keys[base + cursor] = key;
    ids[base + cursor] = id;
    cursor += 1;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void writeKeys(uint3 tid : SV_DispatchThreadID) {
    if (int(tid.x) >= pc.count) return;
    Splat2D s = splats[tid.x];
    int tx0, tx1, ty0, ty1;
    float xMin, xMax, yMin, yMax, a, b, c, t, xMag, yMag;
    if (tileRows(s, pc.tilesX, pc.tilesY, tx0, tx1, ty0, ty1, xMin, xMax, yMin, yMax, a, b, c, t, xMag, yMag) == 0) return;
    float2 mean = s.xyDepthRadius.xy;
    int cursor = 0;
    int base = offsets[tid.x];
    uint id = tid.x;
    if ((ty1 - ty0) <= (tx1 - tx0)) {
        for (int ty = ty0; ty <= ty1; ++ty) {
            float y0 = max(float(ty * kTile), yMin);
            float y1 = min(float((ty + 1) * kTile), yMax);
            float lo, hi;
            if (!spanX(mean, a, b, c, t, xMag, y0, y1, lo, hi)) continue;
            int a0 = tileOf(lo, pc.tilesX);
            int a1 = tileOf(hi, pc.tilesX);
            for (int tx = a0; tx <= a1; ++tx) emit(base, cursor, ty * pc.tilesX + tx, s.xyDepthRadius.z, id);
        }
    } else {
        for (int tx = tx0; tx <= tx1; ++tx) {
            float x0 = max(float(tx * kTile), xMin);
            float x1 = min(float((tx + 1) * kTile), xMax);
            float lo, hi;
            if (!spanY(mean, a, b, c, t, yMag, x0, x1, lo, hi)) continue;
            int b0 = tileOf(lo, pc.tilesY);
            int b1 = tileOf(hi, pc.tilesY);
            for (int ty = b0; ty <= b1; ++ty) emit(base, cursor, ty * pc.tilesX + tx, s.xyDepthRadius.z, id);
        }
    }
}
