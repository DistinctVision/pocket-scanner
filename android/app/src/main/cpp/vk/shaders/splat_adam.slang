#include "splat_common.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<Gauss> value;

[[vk::binding(1, 0)]]
StructuredBuffer<Gauss> grad;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Gauss> moment;

[[vk::binding(3, 0)]]
RWStructuredBuffer<Gauss> vel;

struct AdamPush {
    int count;
    int step;
    float lrMean;
    float lrColor;
    float lrOp;
    float lrQuat;
    float lrScale;
    float pad;
};

[[vk::push_constant]]
AdamPush pc;

float adam1(float v, float g, inout float m, inout float vel, float lr, float b1, float b2, float bc1, float bc2) {
    m = b1 * m + (1.0 - b1) * g;
    vel = b2 * vel + (1.0 - b2) * g * g;
    float mh = m / bc1;
    float vh = vel / bc2;
    return v - lr * mh / (sqrt(vh) + 1e-8);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void adamGauss(uint3 tid : SV_DispatchThreadID) {
    if (int(tid.x) >= pc.count) return;
    Gauss v = value[tid.x];
    Gauss g = grad[tid.x];
    Gauss m = moment[tid.x];
    Gauss s = vel[tid.x];
    float b1 = 0.9;
    float b2 = 0.999;
    float bc1 = 1.0 - pow(b1, float(pc.step));
    float bc2 = 1.0 - pow(b2, float(pc.step));
    [unroll]
    for (int k = 0; k < 3; ++k) {
        v.meanOp[k] = adam1(v.meanOp[k], g.meanOp[k], m.meanOp[k], s.meanOp[k], pc.lrMean, b1, b2, bc1, bc2);
        v.colorPad[k] = adam1(v.colorPad[k], g.colorPad[k], m.colorPad[k], s.colorPad[k], pc.lrColor, b1, b2, bc1, bc2);
        v.scalePad[k] = adam1(v.scalePad[k], g.scalePad[k], m.scalePad[k], s.scalePad[k], pc.lrScale, b1, b2, bc1, bc2);
    }
    v.meanOp.w = adam1(v.meanOp.w, g.meanOp.w, m.meanOp.w, s.meanOp.w, pc.lrOp, b1, b2, bc1, bc2);
    [unroll]
    for (int k = 0; k < 4; ++k) {
        v.quat[k] = adam1(v.quat[k], g.quat[k], m.quat[k], s.quat[k], pc.lrQuat, b1, b2, bc1, bc2);
    }
    v.quat = v.quat * rsqrt(max(dot(v.quat, v.quat), 1e-12));
    v.colorPad.xyz = saturate(v.colorPad.xyz);
    value[tid.x] = v;
    moment[tid.x] = m;
    vel[tid.x] = s;
}
