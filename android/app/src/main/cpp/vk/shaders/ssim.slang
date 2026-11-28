// 11x11 separable SSIM. C1 and C2 match fused-ssim. Channels are independent.
// The backward pass convolves the stored partials with the same kernel.

static const float kC1 = 0.0001;
static const float kC2 = 0.0009;
static const float kW[6] = {0.26601171, 0.21300553, 0.10936069, 0.03600077, 0.00759876, 0.00102838};

[[vk::binding(0, 0)]]
StructuredBuffer<float> rendered;

[[vk::binding(1, 0)]]
StructuredBuffer<float> target;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> stats;

[[vk::binding(3, 0)]]
StructuredBuffer<float> upstream;

[[vk::binding(4, 0)]]
RWStructuredBuffer<float> imageGrad;

struct SsimPush {
    int width;
    int height;
    int backward;
    float scale;
};

[[vk::push_constant]]
SsimPush pc;

float pix(StructuredBuffer<float> img, int x, int y, int c) {
    if (x < 0 || y < 0 || x >= pc.width || y >= pc.height) return 0.0;
    return img[(y * pc.width + x) * 3 + c];
}

float convAt(StructuredBuffer<float> img, int x, int y, int c) {
    float acc = 0;
    for (int oy = -5; oy <= 5; ++oy) {
        float wy = oy < 0 ? kW[-oy] : kW[oy];
        for (int ox = -5; ox <= 5; ++ox) {
            float wx = ox < 0 ? kW[-ox] : kW[ox];
            acc += wy * wx * pix(img, x + ox, y + oy, c);
        }
    }
    return acc;
}

float4 ssimStatistics(float mu1, float mu2, float x2, float y2, float xy) {
    float variance1 = x2 - mu1 * mu1;
    float s1 = max(variance1, 0.0);
    float s2 = max(y2 - mu2 * mu2, 0.0);
    float s12 = xy - mu1 * mu2;
    float A = 2.0 * mu1 * mu2 + kC1;
    float B = mu1 * mu1 + mu2 * mu2 + kC1;
    float C = 2.0 * s12 + kC2;
    float D = s1 + s2 + kC2;
    float den = B * D;
    float ssim = A * C / den;
    float dMu = ((2.0 * mu2 * C) * den - A * C * (2.0 * mu1 * D)) / (den * den);
    float dS1 = variance1 > 0.0 ? -(A * C * B) / (den * den) : 0.0;
    float dS12 = (2.0 * A * den) / (den * den);
    dMu += dS1 * (-2.0 * mu1) + dS12 * (-mu2);
    return float4(ssim, dMu, dS1, dS12);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void ssimForward(uint3 tid : SV_DispatchThreadID) {
    if (int(tid.x) >= pc.width || int(tid.y) >= pc.height) return;
    int x = int(tid.x);
    int y = int(tid.y);
    int pixId = y * pc.width + x;
    for (int c = 0; c < 3; ++c) {
        float mu1 = convAt(rendered, x, y, c);
        float mu2 = convAt(target, x, y, c);
        float x2 = 0;
        float y2 = 0;
        float xy = 0;
        for (int oy = -5; oy <= 5; ++oy) {
            float wy = oy < 0 ? kW[-oy] : kW[oy];
            for (int ox = -5; ox <= 5; ++ox) {
                float wx = ox < 0 ? kW[-ox] : kW[ox];
                float a = pix(rendered, x + ox, y + oy, c);
                float b = pix(target, x + ox, y + oy, c);
                float w = wy * wx;
                x2 += w * a * a;
                y2 += w * b * b;
                xy += w * a * b;
            }
        }
        float4 value = ssimStatistics(mu1, mu2, x2, y2, xy);
        int base = (pixId * 3 + c) * 4;
        stats[base] = value.x;
        stats[base + 1] = value.y;
        stats[base + 2] = value.z;
        stats[base + 3] = value.w;
    }
}

[[vk::binding(7, 0)]]
RWStructuredBuffer<float> rowSums;

[shader("compute")]
[numthreads(16, 8, 1)]
void ssimForwardRows(uint3 tid : SV_DispatchThreadID) {
    int x = int(tid.x);
    int y = int(tid.y);
    if (x >= pc.width || y >= pc.height) return;
    for (int c = 0; c < 3; ++c) {
        float4 moments = 0;
        float product = 0;
        [unroll]
        for (int dx = -5; dx <= 5; ++dx) {
            if (x + dx < 0 || x + dx >= pc.width) continue;
            int i = (y * pc.width + x + dx) * 3 + c;
            float a = rendered[i];
            float b = target[i];
            float w = kW[abs(dx)];
            moments += w * float4(a, b, a * a, b * b);
            product += w * a * b;
        }
        int i = ((y * pc.width + x) * 3 + c) * 5;
        rowSums[i] = moments.x;
        rowSums[i + 1] = moments.y;
        rowSums[i + 2] = moments.z;
        rowSums[i + 3] = moments.w;
        rowSums[i + 4] = product;
    }
}

[shader("compute")]
[numthreads(16, 8, 1)]
void ssimForwardColumns(uint3 tid : SV_DispatchThreadID) {
    int x = int(tid.x);
    int y = int(tid.y);
    if (x >= pc.width || y >= pc.height) return;
    for (int c = 0; c < 3; ++c) {
        float4 moments = 0;
        float product = 0;
        [unroll]
        for (int dy = -5; dy <= 5; ++dy) {
            if (y + dy < 0 || y + dy >= pc.height) continue;
            int i = (((y + dy) * pc.width + x) * 3 + c) * 5;
            float w = kW[abs(dy)];
            moments += w * float4(rowSums[i], rowSums[i + 1], rowSums[i + 2], rowSums[i + 3]);
            product += w * rowSums[i + 4];
        }
        float4 value = ssimStatistics(moments.x, moments.y, moments.z, moments.w, product);
        int base = ((y * pc.width + x) * 3 + c) * 4;
        stats[base] = value.x;
        stats[base + 1] = value.y;
        stats[base + 2] = value.z;
        stats[base + 3] = value.w;
    }
}

[shader("compute")]
[numthreads(16, 8, 1)]
void ssimBackwardRows(uint3 tid : SV_DispatchThreadID) {
    int x = int(tid.x);
    int y = int(tid.y);
    if (x >= pc.width || y >= pc.height) return;
    for (int c = 0; c < 3; ++c) {
        float3 partial = 0;
        [unroll]
        for (int dx = -5; dx <= 5; ++dx) {
            if (x + dx < 0 || x + dx >= pc.width) continue;
            int i = ((y * pc.width + x + dx) * 3 + c) * 4;
            partial += kW[abs(dx)] * float3(stats[i + 1], stats[i + 2], stats[i + 3]);
        }
        int i = ((y * pc.width + x) * 3 + c) * 3;
        rowSums[i] = partial.x;
        rowSums[i + 1] = partial.y;
        rowSums[i + 2] = partial.z;
    }
}

[shader("compute")]
[numthreads(16, 8, 1)]
void ssimBackwardColumns(uint3 tid : SV_DispatchThreadID) {
    int x = int(tid.x);
    int y = int(tid.y);
    if (x >= pc.width || y >= pc.height) return;
    for (int c = 0; c < 3; ++c) {
        float3 grad = 0;
        [unroll]
        for (int dy = -5; dy <= 5; ++dy) {
            if (y + dy < 0 || y + dy >= pc.height) continue;
            int i = (((y + dy) * pc.width + x) * 3 + c) * 3;
            grad += kW[abs(dy)] * float3(rowSums[i], rowSums[i + 1], rowSums[i + 2]);
        }
        int pixId = (y * pc.width + x) * 3 + c;
        float a = rendered[pixId];
        float b = target[pixId];
        float inv = 1.0 / float(pc.width * pc.height * 3);
        float l1 = a > b ? inv : (a < b ? -inv : 0.0);
        imageGrad[pixId] = pc.scale * (grad.x + grad.y * 2.0 * a + grad.z * b) + l1;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void ssimBackward(uint3 tid : SV_DispatchThreadID) {
    if (int(tid.x) >= pc.width || int(tid.y) >= pc.height) return;
    int x = int(tid.x);
    int y = int(tid.y);
    float3 grad = 0;
    for (int c = 0; c < 3; ++c) {
        float accMu = 0;
        float accS1 = 0;
        float accS12 = 0;
        for (int oy = -5; oy <= 5; ++oy) {
            float wy = oy < 0 ? kW[-oy] : kW[oy];
            for (int ox = -5; ox <= 5; ++ox) {
                float wx = ox < 0 ? kW[-ox] : kW[ox];
                int sx = x + ox;
                int sy = y + oy;
                if (sx < 0 || sy < 0 || sx >= pc.width || sy >= pc.height) continue;
                int base = ((sy * pc.width + sx) * 3 + c) * 4;
                float up = pc.scale;
                float w = wy * wx * up;
                accMu += w * stats[base + 1];
                accS1 += w * stats[base + 2];
                accS12 += w * stats[base + 3];
            }
        }
        float a = pix(rendered, x, y, c);
        float b = pix(target, x, y, c);
        float inv = 1.0 / float(pc.width * pc.height * 3);
        float l1 = a > b ? inv : (a < b ? -inv : 0.0);
        grad[c] = accMu + accS1 * 2.0 * a + accS12 * b + l1;
    }
    int pixId = y * pc.width + x;
    imageGrad[pixId * 3] = grad.x;
    imageGrad[pixId * 3 + 1] = grad.y;
    imageGrad[pixId * 3 + 2] = grad.z;
}

[[vk::binding(5, 0)]]
StructuredBuffer<float> corrected;

[[vk::binding(6, 0)]]
RWStructuredBuffer<float> partial;

void reduceColorLoss(int lane, int stride) {
    float l1 = 0;
    float mse = 0;
    float ssim = 0;
    int n = pc.width * pc.height;
    for (int p = lane; p < n; p += stride) {
        for (int c = 0; c < 3; ++c) {
            float d = corrected[p * 3 + c] - target[p * 3 + c];
            l1 += abs(d);
            mse += d * d;
            ssim += stats[(p * 3 + c) * 4];
        }
    }
    partial[lane * 3] = l1;
    partial[lane * 3 + 1] = mse;
    partial[lane * 3 + 2] = ssim;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void reduceLoss(uint3 tid : SV_DispatchThreadID) {
    reduceColorLoss(int(tid.x), 256);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void reduceLossWide(uint3 tid : SV_DispatchThreadID) {
    reduceColorLoss(int(tid.x), 8192);
}
