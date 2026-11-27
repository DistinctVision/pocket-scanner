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
RWStructuredBuffer<uint> accum;

[[vk::binding(5, 0)]]
StructuredBuffer<PixelBlend> pixels;

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
groupshared uint tileGrad[128 * 10];

void addLocal(int slot, int param, float value) {
    if (value == 0.0) return;
    int index = slot * 10 + param;
    uint previous;
    InterlockedOr(tileGrad[index], 0u, previous);
    uint expected;
    do {
        expected = previous;
        InterlockedCompareExchange(tileGrad[index], expected,
                                   asuint(asfloat(expected) + value), previous);
    } while (previous != expected);
}

void shadeBack<let kMode : int>(int x, int y, PixelBlend pixel, inout float T,
                                inout float3 behind, inout float behindZ, int base, int n) {
    n = min(n, pixel.end - base);
    if (n <= 0) return;
    int pix = y * pc.width + x;
    float3 dC = 0;
    float dZpix = 0;
    // Read in every specialization so the push block stays the same size.
    float weight = pc.depthWeight;
    if (kMode != 1) dC = float3(imageGrad[pix * 3], imageGrad[pix * 3 + 1], imageGrad[pix * 3 + 2]);
    if (kMode != 0) dZpix = depthGrad[pix] * weight / (1.0 - pixel.transmittance);
    float2 p = float2(float(x), float(y)) + 0.5;
    for (int i = n - 1; i >= 0; --i) {
        Splat2D s = batch[i];
        float2 d = p - s.xyDepthRadius.xy;
        float G;
        float a = splatAlpha(s, d, G);
        if (a < kMinAlpha) continue;
        float Tbefore = T / (1.0 - a);
        float dla = 0;
        float3 dColor = 0;
        float dDepth = 0;
        if (kMode != 1) {
            float3 dA = (s.colorAlive.xyz - behind) * Tbefore;
            dla += dot(dC, dA);
            dColor = dC * (a * Tbefore);
        }
        // Shifting every contributing depth by the final expected depth
        // incorporates the derivative of the alpha denominator.
        float z = s.xyDepthRadius.z - pixel.depth;
        if (kMode != 0) {
            dla += dZpix * (z - behindZ) * Tbefore;
            dDepth = dZpix * (a * Tbefore);
        }
        float sig = s.conicAlpha.w;
        if (sig * G < kMaxAlpha) {
            float dQ = -0.5 * dla * sig * G;
            float ddx = dQ * (2.0 * s.conicAlpha.x * d.x + 2.0 * s.conicAlpha.y * d.y);
            float ddy = dQ * (2.0 * s.conicAlpha.y * d.x + 2.0 * s.conicAlpha.z * d.y);
            addLocal(i, 0, -ddx);
            addLocal(i, 1, -ddy);
            addLocal(i, 2, dQ * d.x * d.x);
            addLocal(i, 3, dQ * 2.0 * d.x * d.y);
            addLocal(i, 4, dQ * d.y * d.y);
            addLocal(i, 8, dla * G * sig * (1.0 - sig));
        }
        if (kMode != 1) {
            addLocal(i, 5, dColor.x);
            addLocal(i, 6, dColor.y);
            addLocal(i, 7, dColor.z);
        }
        if (kMode != 0) addLocal(i, 9, dDepth);
        T = Tbefore;
        if (kMode != 1) behind = s.colorAlive.xyz * a + behind * (1.0 - a);
        if (kMode != 0) behindZ = z * a + behindZ * (1.0 - a);
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
    PixelBlend pixel0 = (PixelBlend)0;
    PixelBlend pixel1 = (PixelBlend)0;
    if (x0 < pc.width && y0 < pc.height) pixel0 = pixels[y0 * pc.width + x0];
    if (x1 < pc.width && y1 < pc.height) pixel1 = pixels[y1 * pc.width + x1];
    float T0 = pixel0.transmittance;
    float T1 = pixel1.transmittance;
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
        shadeBack<kMode>(x0, y0, pixel0, T0, behind0, behindZ0, cursor, n);
        shadeBack<kMode>(x1, y1, pixel1, T1, behind1, behindZ1, cursor, n);
        GroupMemoryBarrierWithGroupSync();
        if (int(lane) < n) {
            uint id = batchId[lane];
            [unroll]
            for (int param = 0; param < kParams; ++param) {
                addFloat(accum, int(id) * 12 + param, asfloat(tileGrad[int(lane) * 10 + param]));
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
