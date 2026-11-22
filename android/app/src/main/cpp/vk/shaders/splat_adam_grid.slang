static const float kGradScale = 100000.0;

[[vk::binding(0, 0)]]
RWStructuredBuffer<float> gridValue;

[[vk::binding(1, 0)]]
StructuredBuffer<int> gridGrad;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> gridMoment;

[[vk::binding(3, 0)]]
RWStructuredBuffer<float> gridVel;

struct GridAdamPush {
    int offset;
    int count;
    int step;
    float lr;
    float beta1;
    float beta2;
    float eps;
};

[[vk::push_constant]]
GridAdamPush pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void adamGrid(uint3 tid : SV_DispatchThreadID) {
    if (int(tid.x) >= uint(pc.count)) return;
    int i = pc.offset + int(tid.x);
    float g = float(gridGrad[i]) / kGradScale;
    float m = gridMoment[i] * pc.beta1 + (1.0 - pc.beta1) * g;
    float s = gridVel[i] * pc.beta2 + (1.0 - pc.beta2) * g * g;
    float bc1 = 1.0 - pow(pc.beta1, float(pc.step));
    float bc2 = 1.0 - pow(pc.beta2, float(pc.step));
    gridValue[i] = gridValue[i] - pc.lr * (m / bc1) / (sqrt(s / bc2) + pc.eps);
    gridMoment[i] = m;
    gridVel[i] = s;
}
