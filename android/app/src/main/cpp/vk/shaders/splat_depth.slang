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

[shader("compute")]
[numthreads(256, 1, 1)]
void depthLoss(uint3 tid : SV_DispatchThreadID) {
    float sum = 0;
    float hits = 0;
    int n = pc.width * pc.height;
    for (int p = int(tid.x); p < n; p += 256) {
        float pred = pixels[p].depth;
        float gt = depthGt[p];
        float grad = 0;
        if (pred > pc.depthMin && gt > pc.depthMin) {
            float invP = 1.0 / pred;
            float invG = 1.0 / gt;
            float diff = invP - invG;
            sum += abs(diff);
            hits += 1.0;
            float sign = diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
            grad = sign * (-invP * invP);
        }
        depthGrad[p] = grad;
    }
    partial[int(tid.x) * 2] = sum;
    partial[int(tid.x) * 2 + 1] = hits;
}
