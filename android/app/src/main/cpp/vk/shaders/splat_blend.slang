#include "splat_common.slang"

// kMode 0 is RGB, 1 is expected depth, 2 is both. Each entry point is its own specialization.
[[vk::binding(0, 0)]]
StructuredBuffer<Splat2D> splats;

[[vk::binding(1, 0)]]
RWStructuredBuffer<int> ranges;

[[vk::binding(2, 0)]]
StructuredBuffer<uint> ids;

[[vk::binding(3, 0)]]
RWStructuredBuffer<float> image;

[[vk::binding(4, 0)]]
RWStructuredBuffer<PixelBlend> pixels;

struct BlendPush {
    int tilesX;
    int tilesY;
    int width;
    int height;
};

[[vk::push_constant]]
BlendPush pc;

groupshared Splat2D batch[128];
groupshared int alive;

void shade<let kMode : int>(int x, int y, inout float3 color, inout PixelBlend pixel, int base, int n) {
    if (x >= pc.width || y >= pc.height || pixel.transmittance <= kMinTransmittance) return;
    float2 pix = float2(float(x), float(y)) + 0.5;
    for (int i = 0; i < n; ++i) {
        if (pixel.transmittance <= kMinTransmittance) break;
        Splat2D s = batch[i];
        float gaussian;
        float a = splatAlpha(s, pix - s.xyDepthRadius.xy, gaussian);
        if (a < kMinAlpha) continue;
        float w = a * pixel.transmittance;
        if (kMode != 1) color += w * s.colorAlive.xyz;
        if (kMode != 0) pixel.depth += w * s.xyDepthRadius.z;
        pixel.transmittance *= 1.0 - a;
        pixel.end = base + i + 1;
    }
}

void writePixel<let kMode : int>(int x, int y, float3 color, PixelBlend pixel) {
    if (x >= pc.width || y >= pc.height) return;
    int index = y * pc.width + x;
    if (kMode != 1) {
        image[index * 3 + 0] = color.x;
        image[index * 3 + 1] = color.y;
        image[index * 3 + 2] = color.z;
    }
    if (kMode != 0 && pixel.end > 0) pixel.depth /= 1.0 - pixel.transmittance;
    pixels[index] = pixel;
}

void blendOne<let kMode : int>(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
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
    PixelBlend pixel0 = {1.0, 0, 0.0};
    PixelBlend pixel1 = pixel0;
    for (int base = begin; base < end; base += 128) {
        int n = min(128, end - base);
        if (int(lane) < n) {
            batch[lane] = splats[ids[base + int(lane)]];
        }
        if (lane == 0) alive = 0;
        GroupMemoryBarrierWithGroupSync();
        shade<kMode>(x0, y0, color0, pixel0, base, n);
        shade<kMode>(x1, y1, color1, pixel1, base, n);
        int live = 0;
        if (x0 < pc.width && y0 < pc.height && pixel0.transmittance > kMinTransmittance) live = 1;
        if (x1 < pc.width && y1 < pc.height && pixel1.transmittance > kMinTransmittance) live = 1;
        if (live != 0) InterlockedOr(alive, 1);
        GroupMemoryBarrierWithGroupSync();
        bool done = alive == 0;
        GroupMemoryBarrierWithGroupSync();
        if (done) {
            end = base + n;
            break;
        }
    }
    if (lane == 0) ranges[tile * 2 + 1] = end;
    writePixel<kMode>(x0, y0, color0, pixel0);
    writePixel<kMode>(x1, y1, color1, pixel1);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void blendRgb(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) { blendOne<0>(gid, gtid); }

[shader("compute")]
[numthreads(128, 1, 1)]
void blendEd(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) { blendOne<1>(gid, gtid); }

[shader("compute")]
[numthreads(128, 1, 1)]
void blendRgbEd(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) { blendOne<2>(gid, gtid); }
