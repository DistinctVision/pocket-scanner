// Shared layout for the splatting kernels. One gaussian is a thin disc
// seeded from a mesh vertex. Scales are stored as logarithms, opacity as a logit.

static const int kTile = 16;
static const int kGridL = 8;
static const int kGridH = 16;
static const int kGridW = 16;
static const int kGridCell = 12 * kGridL * kGridH * kGridW;
static const float kMinAlpha = 1.0 / 255.0;
static const float kMaxAlpha = 0.999;
static const float kMinTransmittance = 1.0 / 255.0;

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

struct PixelBlend {
    float transmittance;
    int end; // Exclusive end in the sorted intersections; zero for an empty pixel.
    float depth;
};

float splatAlpha(Splat2D s, float2 delta, out float gaussian) {
    float q = s.conicAlpha.x * delta.x * delta.x +
              2.0 * s.conicAlpha.y * delta.x * delta.y +
              s.conicAlpha.z * delta.y * delta.y;
    gaussian = q >= 0.0 && q <= 12.0 ? exp(-0.5 * q) : 0.0;
    return min(kMaxAlpha, s.conicAlpha.w * gaussian);
}

// Add a float through its uint bits using the integer atomics of Vulkan 1.1.
void addFloat(RWStructuredBuffer<uint> dest, int index, float value) {
    if (value == 0.0) return;
    uint previous;
    InterlockedOr(dest[index], 0u, previous);
    uint expected;
    do {
        expected = previous;
        InterlockedCompareExchange(dest[index], expected,
                                   asuint(asfloat(expected) + value), previous);
    } while (previous != expected);
}

struct Key64 {
    uint depth;
    uint tile;
};

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

float3x3 cameraCovariance(Gauss g, float3x3 worldToCamera) {
    float3 s = exp(clamp(g.scalePad.xyz, -12.0, 4.0));
    float3x3 scale = float3x3(s.x, 0, 0, 0, s.y, 0, 0, 0, s.z);
    float3x3 transform = mul(mul(worldToCamera, quatMat(g.quat)), scale);
    return mul(transform, transpose(transform));
}

bool projectConic(float3x3 covariance, float3 cam, float2 focal, out float3 conic) {
    float invZ = 1.0 / cam.z;
    float2 f = focal * invZ;
    float3 jx = float3(f.x, 0.0, -f.x * cam.x * invZ);
    float3 jy = float3(0.0, f.y, -f.y * cam.y * invZ);
    float3 covX = mul(covariance, jx);
    float3 covY = mul(covariance, jy);
    float xx = max(dot(jx, covX), 0.0) + 0.3;
    float xy = dot(jx, covY);
    float yy = max(dot(jy, covY), 0.0) + 0.3;
    float det = xx * yy - xy * xy;
    conic = float3(yy, -xy, xx) / max(det, 1e-6);
    return det >= 1e-6;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-clamp(x, -20.0, 20.0)));
}

// Inverse-covariance ellipse a xd^2 + 2 b xd yd + c yd^2 = t, t = 2 log(255 opacity).
bool ellipseBox(float a, float b, float c, float t, out float xMag, out float yMag) {
    xMag = 0.0;
    yMag = 0.0;
    float det = a * c - b * b;
    if (t <= 0.0 || det <= 1e-8 || a <= 1e-8 || c <= 1e-8) return false;
    xMag = sqrt(c * t / det);
    yMag = sqrt(a * t / det);
    return true;
}

bool crossX(float a, float b, float c, float t, float yd, out float xd0, out float xd1) {
    xd0 = 0.0;
    xd1 = 0.0;
    float A = a;
    float B = 2.0 * b * yd;
    float C = c * yd * yd - t;
    float disc = B * B - 4.0 * A * C;
    if (disc < 0.0 || abs(A) <= 1e-12) return false;
    float s = sqrt(disc);
    xd0 = (-B - s) / (2.0 * A);
    xd1 = (-B + s) / (2.0 * A);
    return true;
}

bool crossY(float a, float b, float c, float t, float xd, out float yd0, out float yd1) {
    yd0 = 0.0;
    yd1 = 0.0;
    float A = c;
    float B = 2.0 * b * xd;
    float C = a * xd * xd - t;
    float disc = B * B - 4.0 * A * C;
    if (disc < 0.0 || abs(A) <= 1e-12) return false;
    float s = sqrt(disc);
    yd0 = (-B - s) / (2.0 * A);
    yd1 = (-B + s) / (2.0 * A);
    return true;
}

bool spanX(float2 mean, float a, float b, float c, float t, float xMag, float y0, float y1, out float lo, out float hi) {
    lo = 1e30;
    hi = -1e30;
    float xd0;
    float xd1;
    if (crossX(a, b, c, t, y0 - mean.y, xd0, xd1)) {
        lo = min(lo, mean.x + min(xd0, xd1));
        hi = max(hi, mean.x + max(xd0, xd1));
    }
    if (crossX(a, b, c, t, y1 - mean.y, xd0, xd1)) {
        lo = min(lo, mean.x + min(xd0, xd1));
        hi = max(hi, mean.x + max(xd0, xd1));
    }
    float yLeft = mean.y + (-b * (-xMag) / c);
    float yRight = mean.y + (-b * xMag / c);
    if (yLeft >= y0 && yLeft <= y1) lo = min(lo, mean.x - xMag);
    if (yRight >= y0 && yRight <= y1) hi = max(hi, mean.x + xMag);
    return hi >= lo;
}

bool spanY(float2 mean, float a, float b, float c, float t, float yMag, float x0, float x1, out float lo, out float hi) {
    lo = 1e30;
    hi = -1e30;
    float yd0;
    float yd1;
    if (crossY(a, b, c, t, x0 - mean.x, yd0, yd1)) {
        lo = min(lo, mean.y + min(yd0, yd1));
        hi = max(hi, mean.y + max(yd0, yd1));
    }
    if (crossY(a, b, c, t, x1 - mean.x, yd0, yd1)) {
        lo = min(lo, mean.y + min(yd0, yd1));
        hi = max(hi, mean.y + max(yd0, yd1));
    }
    float xBottom = mean.x + (-b * (-yMag) / a);
    float xTop = mean.x + (-b * yMag / a);
    if (xBottom >= x0 && xBottom <= x1) lo = min(lo, mean.y - yMag);
    if (xTop >= x0 && xTop <= x1) hi = max(hi, mean.y + yMag);
    return hi >= lo;
}

int tileOf(float coord, int limit) {
    if (!(coord >= 0.0) || limit <= 0) return 0;
    float scaled = coord / float(kTile);
    if (!(scaled < float(limit))) return limit - 1;
    return int(floor(scaled));
}

int tileRows(Splat2D s, int tilesX, int tilesY, out int tx0, out int tx1, out int ty0, out int ty1,
             out float xMin, out float xMax, out float yMin, out float yMax, out float a, out float b, out float c, out float t, out float xMag, out float yMag) {
    tx0 = 0;
    tx1 = -1;
    ty0 = 0;
    ty1 = -1;
    a = s.conicAlpha.x;
    b = s.conicAlpha.y;
    c = s.conicAlpha.z;
    float alpha = s.conicAlpha.w;
    t = 2.0 * log(max(alpha, 1e-6) * 255.0);
    xMag = 0.0;
    yMag = 0.0;
    xMin = 0.0;
    xMax = 0.0;
    yMin = 0.0;
    yMax = 0.0;
    if (s.colorAlive.w < 0.5 || !ellipseBox(a, b, c, t, xMag, yMag)) return 0;
    float2 mean = s.xyDepthRadius.xy;
    xMin = mean.x - xMag;
    xMax = mean.x + xMag;
    yMin = mean.y - yMag;
    yMax = mean.y + yMag;
    tx0 = tileOf(xMin, tilesX);
    tx1 = tileOf(xMax, tilesX);
    ty0 = tileOf(yMin, tilesY);
    ty1 = tileOf(yMax, tilesY);
    if (tx0 > tx1 || ty0 > ty1) return 0;
    return 1;
}

int countTilesOf(Splat2D s, int tilesX, int tilesY) {
    int tx0, tx1, ty0, ty1;
    float xMin, xMax, yMin, yMax, a, b, c, t, xMag, yMag;
    if (tileRows(s, tilesX, tilesY, tx0, tx1, ty0, ty1, xMin, xMax, yMin, yMax, a, b, c, t, xMag, yMag) == 0) return 0;
    float2 mean = s.xyDepthRadius.xy;
    int n = 0;
    if ((ty1 - ty0) <= (tx1 - tx0)) {
        for (int ty = ty0; ty <= ty1; ++ty) {
            float y0 = max(float(ty * kTile), yMin);
            float y1 = min(float((ty + 1) * kTile), yMax);
            float lo, hi;
            if (!spanX(mean, a, b, c, t, xMag, y0, y1, lo, hi)) continue;
            int a0 = tileOf(lo, tilesX);
            int a1 = tileOf(hi, tilesX);
            if (a0 <= a1) n += a1 - a0 + 1;
        }
    } else {
        for (int tx = tx0; tx <= tx1; ++tx) {
            float x0 = max(float(tx * kTile), xMin);
            float x1 = min(float((tx + 1) * kTile), xMax);
            float lo, hi;
            if (!spanY(mean, a, b, c, t, yMag, x0, x1, lo, hi)) continue;
            int b0 = tileOf(lo, tilesY);
            int b1 = tileOf(hi, tilesY);
            if (b0 <= b1) n += b1 - b0 + 1;
        }
    }
    return n;
}
