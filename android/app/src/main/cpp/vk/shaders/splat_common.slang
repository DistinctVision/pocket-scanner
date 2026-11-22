// Shared layout for the splatting kernels. One gaussian is a thin disc
// seeded from a mesh vertex. Scales are stored as logarithms, opacity as a logit.

static const int kTile = 16;
static const int kBin = 48;
static const int kKeep = 8;
static const int kGridL = 8;
static const int kGridH = 16;
static const int kGridW = 16;
static const int kGridCell = 12 * kGridL * kGridH * kGridW;
static const float kGradScale = 100000.0;

struct Gauss {
    float4 meanOp;
    float4 colorPad;
    float4 quat;
    float4 scalePad;
};

struct Splat2D {
    float4 xyDepthRadius;
    float4 conicAlpha;
    float4 colorAlive;
};

struct Keep {
    int n;
    int id0;
    int id1;
    int id2;
    int id3;
    int id4;
    int id5;
    int id6;
    int id7;
    float a0;
    float a1;
    float a2;
    float a3;
    float a4;
    float a5;
    float a6;
    float a7;
    float t0;
    float t1;
    float t2;
    float t3;
    float t4;
    float t5;
    float t6;
    float t7;
};

int keepId(Keep k, int i) {
    if (i == 0) return k.id0;
    if (i == 1) return k.id1;
    if (i == 2) return k.id2;
    if (i == 3) return k.id3;
    if (i == 4) return k.id4;
    if (i == 5) return k.id5;
    if (i == 6) return k.id6;
    return k.id7;
}

float keepA(Keep k, int i) {
    if (i == 0) return k.a0;
    if (i == 1) return k.a1;
    if (i == 2) return k.a2;
    if (i == 3) return k.a3;
    if (i == 4) return k.a4;
    if (i == 5) return k.a5;
    if (i == 6) return k.a6;
    return k.a7;
}

float keepT(Keep k, int i) {
    if (i == 0) return k.t0;
    if (i == 1) return k.t1;
    if (i == 2) return k.t2;
    if (i == 3) return k.t3;
    if (i == 4) return k.t4;
    if (i == 5) return k.t5;
    if (i == 6) return k.t6;
    return k.t7;
}

float3 quatRotate(float4 q, float3 v) {
    float3 u = q.xyz;
    return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

float3x3 quatMat(float4 q) {
    float4 n = q * rsqrt(max(dot(q, q), 1e-12));
    float x = n.x;
    float y = n.y;
    float z = n.z;
    float w = n.w;
    return float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y));
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-clamp(x, -20.0, 20.0)));
}
