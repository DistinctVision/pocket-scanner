#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Splat2D> splats;

[[vk::binding(1, 0)]]
StructuredBuffer<Keep> kept;

[[vk::binding(2, 0)]]
StructuredBuffer<float> imageGrad;

[[vk::binding(3, 0)]]
RWStructuredBuffer<int> accum;

[[vk::binding(4, 0)]]
StructuredBuffer<int> counts;

[[vk::binding(5, 0)]]
StructuredBuffer<int> bins;

struct BackPush {
    int width;
    int height;
    int tilesX;
    int pad1;
};

[[vk::push_constant]]
BackPush pc;

groupshared int batchId[kBin];
groupshared Splat2D batch[kBin];
// Nine parameters per splat in this tile. Pixels add here; one global atomic flushes each sum.
groupshared int tileGrad[kBin * 9];

int quantize(float value) {
    return int(clamp(value * kGradScale, -2000000000.0, 2000000000.0));
}

void addLocal(int slot, int param, float value) {
    int q = quantize(value);
    if (q != 0) InterlockedAdd(tileGrad[slot * 9 + param], q);
}

int batchSlot(int id, int n) {
    for (int i = 0; i < n; ++i) {
        if (batchId[i] == id) return i;
    }
    return -1;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void blendBackward(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint lane = gtid.x;
    int tile = int(gid.y) * pc.tilesX + int(gid.x);
    int n = min(counts[tile], kBin);
    if (int(lane) < n) {
        int id = bins[tile * kBin + int(lane)];
        batchId[lane] = id;
        batch[lane] = splats[id];
    }
    for (int i = int(lane); i < kBin * 9; i += 128) tileGrad[i] = 0;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (int sub = 0; sub < 2; ++sub) {
        int local = int(lane) + sub * 128;
        int x = int(gid.x) * kTile + (local & 15);
        int y = int(gid.y) * kTile + (local >> 4);
        if (x >= pc.width || y >= pc.height) continue;
        int pix = y * pc.width + x;
        Keep k = kept[pix];
        if (k.n <= 0) continue;
        float3 dC = float3(imageGrad[pix * 3], imageGrad[pix * 3 + 1], imageGrad[pix * 3 + 2]);
        float2 p = float2(float(x), float(y)) + 0.5;
        float3 behind = 0;
        for (int step = k.n - 1; step >= 0; --step) {
            int id = keepId(k, step);
            int slot = batchSlot(id, n);
            float a = keepA(k, step);
            float T = keepT(k, step);
            Splat2D s = slot >= 0 ? batch[slot] : splats[id];
            float3 dA = (s.colorAlive.xyz - behind) * T;
            float dla = dot(dC, dA);
            float3 dColor = dC * (a * T);
            float2 d = p - s.xyDepthRadius.xy;
            float q = s.conicAlpha.x * d.x * d.x + 2.0 * s.conicAlpha.y * d.x * d.y + s.conicAlpha.z * d.y * d.y;
            float G = exp(-0.5 * q);
            float sig = max(s.conicAlpha.w, 1e-6) / max(G, 1e-6);
            sig = clamp(sig, 0.0, 1.0);
            float dG = dla * sig;
            float dLogit = dla * G * sig * (1.0 - sig);
            float dQ = dG * G * (-0.5);
            float ddx = dQ * (2.0 * s.conicAlpha.x * d.x + 2.0 * s.conicAlpha.y * d.y);
            float ddy = dQ * (2.0 * s.conicAlpha.y * d.x + 2.0 * s.conicAlpha.z * d.y);
            if (slot >= 0) {
                addLocal(slot, 0, -ddx);
                addLocal(slot, 1, -ddy);
                addLocal(slot, 2, dQ * d.x * d.x);
                addLocal(slot, 3, dQ * 2.0 * d.x * d.y);
                addLocal(slot, 4, dQ * d.y * d.y);
                addLocal(slot, 5, dColor.x);
                addLocal(slot, 6, dColor.y);
                addLocal(slot, 7, dColor.z);
                addLocal(slot, 8, dLogit);
            } else {
                int lanes[9] = {quantize(-ddx), quantize(-ddy), quantize(dQ * d.x * d.x), quantize(dQ * 2.0 * d.x * d.y),
                                quantize(dQ * d.y * d.y), quantize(dColor.x), quantize(dColor.y), quantize(dColor.z),
                                quantize(dLogit)};
                [unroll]
                for (int param = 0; param < 9; ++param) {
                    if (lanes[param] != 0) InterlockedAdd(accum[id * 12 + param], lanes[param]);
                }
            }
            behind = s.colorAlive.xyz * a + behind * (1.0 - a);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (int(lane) < n) {
        int id = batchId[lane];
        [unroll]
        for (int param = 0; param < 9; ++param) {
            int q = tileGrad[int(lane) * 9 + param];
            if (q != 0) InterlockedAdd(accum[id * 12 + param], q);
        }
    }
}
