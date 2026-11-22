#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Splat2D> splats;

[[vk::binding(1, 0)]]
StructuredBuffer<int> counts;

[[vk::binding(2, 0)]]
StructuredBuffer<int> bins;

[[vk::binding(3, 0)]]
RWStructuredBuffer<float> image;

[[vk::binding(4, 0)]]
RWStructuredBuffer<Keep> kept;

struct BlendPush {
    int tilesX;
    int tilesY;
    int width;
    int height;
};

[[vk::push_constant]]
BlendPush pc;

// The whole 16x16 tile shares one copy of its sorted bin.
groupshared int batchId[kBin];
groupshared Splat2D batch[kBin];

[shader("compute")]
[numthreads(128, 1, 1)]
void blendForward(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint lane = gtid.x;
    int tile = int(gid.y) * pc.tilesX + int(gid.x);
    int n = min(counts[tile], kBin);
    if (int(lane) < n) {
        int id = bins[tile * kBin + int(lane)];
        batchId[lane] = id;
        batch[lane] = splats[id];
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (int sub = 0; sub < 2; ++sub) {
        int local = int(lane) + sub * 128;
        int x = int(gid.x) * kTile + (local & 15);
        int y = int(gid.y) * kTile + (local >> 4);
        if (x >= pc.width || y >= pc.height) continue;
        float2 pix = float2(float(x), float(y)) + 0.5;
        float3 color = 0;
        float T = 1.0;
        Keep k = (Keep)0;
        for (int i = 0; i < n && k.n < kKeep && T > 0.004; ++i) {
            Splat2D s = batch[i];
            float2 d = pix - s.xyDepthRadius.xy;
            float q = s.conicAlpha.x * d.x * d.x + 2.0 * s.conicAlpha.y * d.x * d.y + s.conicAlpha.z * d.y * d.y;
            if (q > 16.0) continue;
            float a = s.conicAlpha.w * exp(-0.5 * q);
            if (a < 1.0 / 255.0) continue;
            color += a * T * s.colorAlive.xyz;
            int id = batchId[i];
            if (k.n == 0) { k.id0 = id; k.a0 = a; k.t0 = T; }
            else if (k.n == 1) { k.id1 = id; k.a1 = a; k.t1 = T; }
            else if (k.n == 2) { k.id2 = id; k.a2 = a; k.t2 = T; }
            else if (k.n == 3) { k.id3 = id; k.a3 = a; k.t3 = T; }
            else if (k.n == 4) { k.id4 = id; k.a4 = a; k.t4 = T; }
            else if (k.n == 5) { k.id5 = id; k.a5 = a; k.t5 = T; }
            else if (k.n == 6) { k.id6 = id; k.a6 = a; k.t6 = T; }
            else { k.id7 = id; k.a7 = a; k.t7 = T; }
            k.n += 1;
            T *= (1.0 - a);
        }
        int pixId = y * pc.width + x;
        image[pixId * 3 + 0] = color.x;
        image[pixId * 3 + 1] = color.y;
        image[pixId * 3 + 2] = color.z;
        kept[pixId] = k;
    }
}
