#include "splat_common.slang"

// Inverse-depth L1 on pixels where both the RealSense depth and the expected depth are valid.
// The stored gradient is d|1/z - 1/gt| / d(z), before the  extent * lambda / N  scale.

[[vk::binding(0, 0)]]
StructuredBuffer<PixelBlend> pixels;

[[vk::binding(1, 0)]]
StructuredBuffer<float> depthGt;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> depthGrad;

[[vk::binding(3, 0)]]
RWStructuredBuffer<float> partial;

struct DepthPush {
    int width;
    int height;
    float depthMin;
    int pad;
};

[[vk::push_constant]]
DepthPush pc;

void reduceDepthLoss<let measureQuality : bool>(int lane, int stride) {
    float sum = 0;
    float hits = 0;
    float measured = 0;
    float absolute = 0;
    float squared = 0;
    int n = pc.width * pc.height;
    for (int p = lane; p < n; p += stride) {
        float pred = pixels[p].depth;
        float gt = depthGt[p];
        float grad = 0;
        bool valid = isfinite(gt) && gt > pc.depthMin;
        if (measureQuality && valid) measured += 1.0;
        if (valid && isfinite(pred) && pred > pc.depthMin) {
            float invP = 1.0 / pred;
            float invG = 1.0 / gt;
            float diff = invP - invG;
            sum += abs(diff);
            hits += 1.0;
            if (measureQuality) {
                float error = pred - gt;
                absolute += abs(error);
                squared += error * error;
            }
            float sign = diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
            grad = sign * (-invP * invP);
        }
        if (!measureQuality) depthGrad[p] = grad;
    }
    int base = lane * (measureQuality ? 5 : 2);
    partial[base] = sum;
    partial[base + 1] = hits;
    if (measureQuality) {
        partial[base + 2] = measured;
        partial[base + 3] = absolute;
        partial[base + 4] = squared;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void depthLoss(uint3 tid : SV_DispatchThreadID) {
    reduceDepthLoss<false>(int(tid.x), 256);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void depthLossWide(uint3 tid : SV_DispatchThreadID) {
    reduceDepthLoss<false>(int(tid.x), 8192);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void depthQualityWide(uint3 tid : SV_DispatchThreadID) {
    reduceDepthLoss<true>(int(tid.x), 8192);
}
