#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<float> grid;

[[vk::binding(1, 0)]]
StructuredBuffer<float> src;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> dst;

[[vk::binding(3, 0)]]
StructuredBuffer<float> dstGrad;

[[vk::binding(4, 0)]]
RWStructuredBuffer<float> srcGrad;

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint> gridGrad;

struct GridPush {
    int width;
    int height;
    int view;
    int views;
    int backward;
    int pad0;
    int pad1;
    int pad2;
    float tv;
};

[[vk::push_constant]]
GridPush pc;

int cellIndex(int view, int c, int l, int y, int x) {
    return view * kGridCell + ((c * kGridL + l) * kGridH + y) * kGridW + x;
}

// Tensor coefficients and their derivative along the guidance axis.
float2 sampleCoef(int view, int c, int3 base, float3 frac) {
    float2 result = 0;
    [unroll]
    for (int iy = 0; iy < 2; ++iy) {
        [unroll]
        for (int ix = 0; ix < 2; ++ix) {
            float w = (ix == 1 ? frac.x : 1 - frac.x) * (iy == 1 ? frac.y : 1 - frac.y);
            int at = cellIndex(view, c, base.z, base.y + iy, base.x + ix);
            float low = grid[at];
            float high = grid[at + kGridW * kGridH];
            result += w * float2(lerp(low, high, frac.z), high - low);
        }
    }
    return result;
}

void splatCoef(int view, int c, int3 base, float3 frac, float grad) {
    [unroll]
    for (int iz = 0; iz < 2; ++iz) {
        [unroll]
        for (int iy = 0; iy < 2; ++iy) {
            [unroll]
            for (int ix = 0; ix < 2; ++ix) {
                float w = (ix == 1 ? frac.x : 1 - frac.x) * (iy == 1 ? frac.y : 1 - frac.y) *
                          (iz == 1 ? frac.z : 1 - frac.z);
                addFloat(gridGrad, cellIndex(view, c, base.z + iz, base.y + iy, base.x + ix), grad * w);
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sliceGrid(uint3 tid : SV_DispatchThreadID) {
    if (int(tid.x) >= pc.width || int(tid.y) >= pc.height) return;
    int pix = int(tid.y) * pc.width + int(tid.x);
    float3 rgb = float3(src[pix * 3], src[pix * 3 + 1], src[pix * 3 + 2]);
    const float3 luma = float3(0.299, 0.587, 0.114);
    float gray = dot(rgb, luma);
    float3 p = float3(
        (float(tid.x) + 0.5) / float(pc.width) * (kGridW - 1),
        (float(tid.y) + 0.5) / float(pc.height) * (kGridH - 1),
        saturate(gray) * (kGridL - 1));
    int3 base = min(int3(p), int3(kGridW - 2, kGridH - 2, kGridL - 2));
    float3 frac = p - float3(base);
    float4 inputColor = float4(rgb, 1.0);
    float3 dRgb = 0;
    float dGray = 0;
    [unroll]
    for (int row = 0; row < 3; ++row) {
        float4 coefficients;
        float4 dGuidance;
        [unroll]
        for (int col = 0; col < 4; ++col) {
            float2 value = sampleCoef(pc.view, row * 4 + col, base, frac);
            coefficients[col] = value.x;
            dGuidance[col] = value.y;
        }
        if (pc.backward == 0) {
            dst[pix * 3 + row] = dot(coefficients, inputColor);
        } else {
            float up = dstGrad[pix * 3 + row];
            dRgb += up * coefficients.xyz;
            dGray += up * dot(dGuidance, inputColor);
            [unroll]
            for (int col = 0; col < 4; ++col) {
                splatCoef(pc.view, row * 4 + col, base, frac, up * inputColor[col]);
            }
        }
    }
    if (pc.backward != 0) {
        if (gray > 0.0 && gray < 1.0) dRgb += dGray * (kGridL - 1) * luma;
        srcGrad[pix * 3] = dRgb.x;
        srcGrad[pix * 3 + 1] = dRgb.y;
        srcGrad[pix * 3 + 2] = dRgb.z;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gridTv(uint3 tid : SV_DispatchThreadID) {
    int cells = kGridCell;
    if (int(tid.x) >= cells) return;
    int local = int(tid.x);
    int x = local % kGridW;
    local /= kGridW;
    int y = local % kGridH;
    local /= kGridH;
    int l = local % kGridL;
    int c = local / kGridL;
    int view = pc.view;
    float v = grid[cellIndex(view, c, l, y, x)];
    const float wx = pc.tv / (12 * kGridL * kGridH * (kGridW - 1));
    const float wy = pc.tv / (12 * kGridL * kGridW * (kGridH - 1));
    const float wl = pc.tv / (12 * (kGridL - 1) * kGridH * kGridW);
    float g = 0;
    if (x + 1 < kGridW) g += wx * sign(v - grid[cellIndex(view, c, l, y, x + 1)]);
    if (y + 1 < kGridH) g += wy * sign(v - grid[cellIndex(view, c, l, y + 1, x)]);
    if (l + 1 < kGridL) g += wl * sign(v - grid[cellIndex(view, c, l + 1, y, x)]);
    if (x > 0) g += wx * sign(v - grid[cellIndex(view, c, l, y, x - 1)]);
    if (y > 0) g += wy * sign(v - grid[cellIndex(view, c, l, y - 1, x)]);
    if (l > 0) g += wl * sign(v - grid[cellIndex(view, c, l - 1, y, x)]);
    int index = cellIndex(view, c, l, y, x);
    // sliceGrid has finished and this invocation owns the coefficient.
    gridGrad[index] = asuint(asfloat(gridGrad[index]) + g);
}
