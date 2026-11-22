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
RWStructuredBuffer<int> gridGrad;

struct GridPush {
    int width;
    int height;
    int view;
    int views;
    int backward;
    int pad0;
    int pad1;
    int pad2;
};

[[vk::push_constant]]
GridPush pc;

int cellIndex(int view, int c, int l, int y, int x) {
    return view * kGridCell + ((c * kGridL + l) * kGridH + y) * kGridW + x;
}

float sampleCoef(int view, int c, float3 p) {
    float3 q = clamp(p, 0.0, float3(kGridW - 1, kGridH - 1, kGridL - 1) - 1e-4);
    int x0 = int(q.x);
    int y0 = int(q.y);
    int l0 = int(q.z);
    int x1 = min(x0 + 1, kGridW - 1);
    int y1 = min(y0 + 1, kGridH - 1);
    int l1 = min(l0 + 1, kGridL - 1);
    float fx = q.x - x0;
    float fy = q.y - y0;
    float fl = q.z - l0;
    float acc = 0;
    [unroll]
    for (int iz = 0; iz < 2; ++iz) {
        [unroll]
        for (int iy = 0; iy < 2; ++iy) {
            [unroll]
            for (int ix = 0; ix < 2; ++ix) {
                float w = (ix == 1 ? fx : 1 - fx) * (iy == 1 ? fy : 1 - fy) * (iz == 1 ? fl : 1 - fl);
                int xx = ix == 1 ? x1 : x0;
                int yy = iy == 1 ? y1 : y0;
                int ll = iz == 1 ? l1 : l0;
                acc += w * grid[cellIndex(view, c, ll, yy, xx)];
            }
        }
    }
    return acc;
}

void splatCoef(int view, int c, float3 p, float grad) {
    float3 q = clamp(p, 0.0, float3(kGridW - 1, kGridH - 1, kGridL - 1) - 1e-4);
    int x0 = int(q.x);
    int y0 = int(q.y);
    int l0 = int(q.z);
    int x1 = min(x0 + 1, kGridW - 1);
    int y1 = min(y0 + 1, kGridH - 1);
    int l1 = min(l0 + 1, kGridL - 1);
    float fx = q.x - x0;
    float fy = q.y - y0;
    float fl = q.z - l0;
    [unroll]
    for (int iz = 0; iz < 2; ++iz) {
        [unroll]
        for (int iy = 0; iy < 2; ++iy) {
            [unroll]
            for (int ix = 0; ix < 2; ++ix) {
                float w = (ix == 1 ? fx : 1 - fx) * (iy == 1 ? fy : 1 - fy) * (iz == 1 ? fl : 1 - fl);
                int xx = ix == 1 ? x1 : x0;
                int yy = iy == 1 ? y1 : y0;
                int ll = iz == 1 ? l1 : l0;
                int qv = int(clamp(grad * w * kGradScale, -2000000000.0, 2000000000.0));
                if (qv != 0) InterlockedAdd(gridGrad[cellIndex(view, c, ll, yy, xx)], qv);
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
    float gray = dot(rgb, float3(0.299, 0.587, 0.114));
    float3 p = float3(
        (float(tid.x) + 0.5) / float(pc.width) * (kGridW - 1),
        (float(tid.y) + 0.5) / float(pc.height) * (kGridH - 1),
        saturate(gray) * (kGridL - 1));
    float m00 = sampleCoef(pc.view, 0, p);
    float m01 = sampleCoef(pc.view, 1, p);
    float m02 = sampleCoef(pc.view, 2, p);
    float m03 = sampleCoef(pc.view, 3, p);
    float m10 = sampleCoef(pc.view, 4, p);
    float m11 = sampleCoef(pc.view, 5, p);
    float m12 = sampleCoef(pc.view, 6, p);
    float m13 = sampleCoef(pc.view, 7, p);
    float m20 = sampleCoef(pc.view, 8, p);
    float m21 = sampleCoef(pc.view, 9, p);
    float m22 = sampleCoef(pc.view, 10, p);
    float m23 = sampleCoef(pc.view, 11, p);
    float3 outc = float3(
        m00 * rgb.x + m01 * rgb.y + m02 * rgb.z + m03,
        m10 * rgb.x + m11 * rgb.y + m12 * rgb.z + m13,
        m20 * rgb.x + m21 * rgb.y + m22 * rgb.z + m23);
    dst[pix * 3] = outc.x;
    dst[pix * 3 + 1] = outc.y;
    dst[pix * 3 + 2] = outc.z;
    if (pc.backward == 0) return;
    float3 up = float3(dstGrad[pix * 3], dstGrad[pix * 3 + 1], dstGrad[pix * 3 + 2]);
    float3 dRgb = float3(
        up.x * m00 + up.y * m10 + up.z * m20,
        up.x * m01 + up.y * m11 + up.z * m21,
        up.x * m02 + up.y * m12 + up.z * m22);
    float dGray = dot(dRgb, float3(0.299, 0.587, 0.114)) * (kGridL - 1);
    srcGrad[pix * 3] = dRgb.x;
    srcGrad[pix * 3 + 1] = dRgb.y;
    srcGrad[pix * 3 + 2] = dRgb.z;
    float dm0 = up.x * rgb.x; float dm1 = up.x * rgb.y; float dm2 = up.x * rgb.z; float dm3 = up.x;
    float dm4 = up.y * rgb.x; float dm5 = up.y * rgb.y; float dm6 = up.y * rgb.z; float dm7 = up.y;
    float dm8 = up.z * rgb.x; float dm9 = up.z * rgb.y; float dm10 = up.z * rgb.z; float dm11 = up.z;
    splatCoef(pc.view, 0, p, dm0);
    splatCoef(pc.view, 1, p, dm1);
    splatCoef(pc.view, 2, p, dm2);
    splatCoef(pc.view, 3, p, dm3);
    splatCoef(pc.view, 4, p, dm4);
    splatCoef(pc.view, 5, p, dm5);
    splatCoef(pc.view, 6, p, dm6);
    splatCoef(pc.view, 7, p, dm7);
    splatCoef(pc.view, 8, p, dm8);
    splatCoef(pc.view, 9, p, dm9);
    splatCoef(pc.view, 10, p, dm10);
    splatCoef(pc.view, 11, p, dm11);
    // Guidance shifts the sample along L. Fold a share of that into the color
    // gradient so a train view can still move the gaussians through the grid.
    srcGrad[pix * 3] += dGray * 0.299;
    srcGrad[pix * 3 + 1] += dGray * 0.587;
    srcGrad[pix * 3 + 2] += dGray * 0.114;
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
    float g = 0;
    if (x + 1 < kGridW) g += sign(v - grid[cellIndex(view, c, l, y, x + 1)]);
    if (y + 1 < kGridH) g += sign(v - grid[cellIndex(view, c, l, y + 1, x)]);
    if (l + 1 < kGridL) g += sign(v - grid[cellIndex(view, c, l + 1, y, x)]);
    if (x > 0) g += sign(v - grid[cellIndex(view, c, l, y, x - 1)]);
    if (y > 0) g += sign(v - grid[cellIndex(view, c, l, y - 1, x)]);
    if (l > 0) g += sign(v - grid[cellIndex(view, c, l - 1, y, x)]);
    int q = int(clamp(g * 0.01 * kGradScale, -2000000000.0, 2000000000.0));
    if (q != 0) InterlockedAdd(gridGrad[cellIndex(view, c, l, y, x)], q);
}
