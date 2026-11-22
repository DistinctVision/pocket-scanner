#include "splat_common.slang"

// kMode 0 is RGB, 1 is expected depth, 2 is both.
[[vk::binding(0, 0)]]
StructuredBuffer<Splat2D> splats;

[[vk::binding(1, 0)]]
StructuredBuffer<int> ranges;

[[vk::binding(2, 0)]]
StructuredBuffer<uint> ids;

[[vk::binding(3, 0)]]
StructuredBuffer<float> imageGrad;

[[vk::binding(4, 0)]]
RWStructuredBuffer<int> accum;

[[vk::binding(5, 0)]]
StructuredBuffer<float> trans;

[[vk::binding(6, 0)]]
StructuredBuffer<float> depthGrad;

struct BackPush {
    int tilesX;
    int tilesY;
    int width;
    int height;
    float depthWeight;
};

[[vk::push_constant]]
BackPush pc;

groupshared uint batchId[128];
groupshared Splat2D batch[128];
groupshared int tileGrad[128 * 10];

int quantize(float value) {
    return int(clamp(value * kGradScale, -2000000000.0, 2000000000.0));
}

void addLocal(int slot, int param, float value) {
    int q = quantize(value);
    if (q != 0) InterlockedAdd(tileGrad[slot * 10 + param], q);
}

void shadeBack<let kMode : int>(int x, int y, inout float T, inout float3 behind, inout float behindZ, int n) {
    if (x >= pc.width || y >= pc.height) return;
    int pix = y * pc.width + x;
    float3 dC = 0;
    float dZpix = 0;
    // Read in every specialization so the push block stays the same size.
    float weight = pc.depthWeight;
    if (kMode != 1) dC = float3(imageGrad[pix * 3], imageGrad[pix * 3 + 1], imageGrad[pix * 3 + 2]);
    if (kMode != 0) dZpix = depthGrad[pix] * weight;
    float2 p = float2(float(x), float(y)) + 0.5;
    for (int i = n - 1; i >= 0; --i) {
        Splat2D s = batch[i];
        float2 d = p - s.xyDepthRadius.xy;
        float q = s.conicAlpha.x * d.x * d.x + 2.0 * s.conicAlpha.y * d.x * d.y + s.conicAlpha.z * d.y * d.y;
        if (q > 12.0) continue;
        float a = s.conicAlpha.w * exp(-0.5 * q);
        if (a < 1.0 / 255.0) continue;
        float Tbefore = T / max(1.0 - a, 1e-4);
        float dla = 0;
        float3 dColor = 0;
        float dDepth = 0;
        if (kMode != 1) {
            float3 dA = (s.colorAlive.xyz - behind) * Tbefore;
            dla += dot(dC, dA);
            dColor = dC * (a * Tbefore);
        }
        if (kMode != 0) {
            float z = s.xyDepthRadius.z;
            dla += dZpix * (z - behindZ) * Tbefore;
            dDepth = dZpix * (a * Tbefore);
        }
        float G = exp(-0.5 * q);
        float sig = clamp(s.conicAlpha.w, 1e-6, 1.0 - 1e-6);
        float dG = dla * sig;
        float dLogit = dla * G * sig * (1.0 - sig);
        float dQ = dG * G * (-0.5);
        float ddx = dQ * (2.0 * s.conicAlpha.x * d.x + 2.0 * s.conicAlpha.y * d.y);
        float ddy = dQ * (2.0 * s.conicAlpha.y * d.x + 2.0 * s.conicAlpha.z * d.y);
        addLocal(i, 0, -ddx);
        addLocal(i, 1, -ddy);
        addLocal(i, 2, dQ * d.x * d.x);
        addLocal(i, 3, dQ * 2.0 * d.x * d.y);
        addLocal(i, 4, dQ * d.y * d.y);
        if (kMode != 1) {
            addLocal(i, 5, dColor.x);
            addLocal(i, 6, dColor.y);
            addLocal(i, 7, dColor.z);
        }
        addLocal(i, 8, dLogit);
        if (kMode != 0) addLocal(i, 9, dDepth);
        T = Tbefore;
        if (kMode != 1) behind = s.colorAlive.xyz * a + behind * (1.0 - a);
        if (kMode != 0) behindZ = s.xyDepthRadius.z * a + behindZ * (1.0 - a);
    }
}

void blendBackOne<let kMode : int>(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
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
    float T0 = (x0 < pc.width && y0 < pc.height) ? trans[y0 * pc.width + x0] : 1.0;
    float T1 = (x1 < pc.width && y1 < pc.height) ? trans[y1 * pc.width + x1] : 1.0;
    float3 behind0 = 0;
    float3 behind1 = 0;
    float behindZ0 = 0;
    float behindZ1 = 0;
    static const int kParams = kMode == 0 ? 9 : 10;
    for (int cursor = end; cursor > begin; ) {
        int n = min(128, cursor - begin);
        cursor -= n;
        if (int(lane) < n) {
            uint id = ids[cursor + int(lane)];
            batchId[lane] = id;
            batch[lane] = splats[id];
        }
        [unroll]
        for (int k = 0; k < 10; ++k) tileGrad[int(lane) * 10 + k] = 0;
        GroupMemoryBarrierWithGroupSync();
        shadeBack<kMode>(x0, y0, T0, behind0, behindZ0, n);
        shadeBack<kMode>(x1, y1, T1, behind1, behindZ1, n);
        GroupMemoryBarrierWithGroupSync();
        if (int(lane) < n) {
            uint id = batchId[lane];
            [unroll]
            for (int param = 0; param < kParams; ++param) {
                int q = tileGrad[int(lane) * 10 + param];
                if (q != 0) InterlockedAdd(accum[id * 12 + param], q);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void blendBackwardRgb(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) { blendBackOne<0>(gid, gtid); }

[shader("compute")]
[numthreads(128, 1, 1)]
void blendBackwardEd(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) { blendBackOne<1>(gid, gtid); }

[shader("compute")]
[numthreads(128, 1, 1)]
void blendBackwardRgbEd(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) { blendBackOne<2>(gid, gtid); }
