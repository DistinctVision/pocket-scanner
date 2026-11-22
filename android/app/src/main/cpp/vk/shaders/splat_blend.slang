#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Splat2D> splats;

[[vk::binding(1, 0)]]
RWStructuredBuffer<int> ranges;

[[vk::binding(2, 0)]]
StructuredBuffer<uint> ids;

[[vk::binding(3, 0)]]
RWStructuredBuffer<float> image;

[[vk::binding(4, 0)]]
RWStructuredBuffer<float> trans;

struct BlendPush {
    int tilesX;
    int tilesY;
    int width;
    int height;
};

[[vk::push_constant]]
BlendPush pc;

groupshared uint batchId[128];
groupshared Splat2D batch[128];
groupshared int alive;

void shade(int x, int y, inout float3 color, inout float T, int n) {
    if (x >= pc.width || y >= pc.height || T <= 1.0 / 255.0) return;
    float2 pix = float2(float(x), float(y)) + 0.5;
    for (int i = 0; i < n; ++i) {
        if (T <= 1.0 / 255.0) break;
        Splat2D s = batch[i];
        float2 d = pix - s.xyDepthRadius.xy;
        float q = s.conicAlpha.x * d.x * d.x + 2.0 * s.conicAlpha.y * d.x * d.y + s.conicAlpha.z * d.y * d.y;
        if (q > 12.0) continue;
        float a = s.conicAlpha.w * exp(-0.5 * q);
        if (a < 1.0 / 255.0) continue;
        color += a * T * s.colorAlive.xyz;
        T *= (1.0 - a);
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void blendForward(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint lane = gtid.x;
    int tile = int(gid.y) * pc.tilesX + int(gid.x);
    int begin = ranges[tile * 2];
    int end = ranges[tile * 2 + 1];
    int local0 = int(lane);
    int local1 = int(lane) + 128;
    int x0 = int(gid.x) * kTile + (local0 & 15);
    int y0 = int(gid.y) * kTile + (local0 >> 4);
    int x1 = int(gid.x) * kTile + (local1 & 15);
    int y1 = int(gid.y) * kTile + (local1 >> 4);
    float3 color0 = 0;
    float3 color1 = 0;
    float T0 = 1.0;
    float T1 = 1.0;
    for (int base = begin; base < end; base += 128) {
        int n = min(128, end - base);
        if (int(lane) < n) {
            uint id = ids[base + int(lane)];
            batchId[lane] = id;
            batch[lane] = splats[id];
        }
        if (lane == 0) alive = 0;
        GroupMemoryBarrierWithGroupSync();
        shade(x0, y0, color0, T0, n);
        shade(x1, y1, color1, T1, n);
        int live = 0;
        if (x0 < pc.width && y0 < pc.height && T0 > 1.0 / 255.0) live = 1;
        if (x1 < pc.width && y1 < pc.height && T1 > 1.0 / 255.0) live = 1;
        if (live != 0) InterlockedOr(alive, 1);
        GroupMemoryBarrierWithGroupSync();
        if (alive == 0) {
            end = base + n;
            break;
        }
    }
    if (lane == 0) ranges[tile * 2 + 1] = end;
    if (x0 < pc.width && y0 < pc.height) {
        int pix = y0 * pc.width + x0;
        image[pix * 3 + 0] = color0.x;
        image[pix * 3 + 1] = color0.y;
        image[pix * 3 + 2] = color0.z;
        trans[pix] = T0;
    }
    if (x1 < pc.width && y1 < pc.height) {
        int pix = y1 * pc.width + x1;
        image[pix * 3 + 0] = color1.x;
        image[pix * 3 + 1] = color1.y;
        image[pix * 3 + 2] = color1.z;
        trans[pix] = T1;
    }
}
