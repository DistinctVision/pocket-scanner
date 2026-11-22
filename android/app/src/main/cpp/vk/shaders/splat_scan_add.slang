[[vk::binding(0, 0)]]
RWStructuredBuffer<int> data;

[[vk::binding(1, 0)]]
StructuredBuffer<int> prefix;

struct AddPush {
    int n;
    int dataBase;
    int prefixBase;
    int pad;
};

[[vk::push_constant]]
AddPush pc;

[shader("compute")]
[numthreads(128, 1, 1)]
void scanAdd(uint3 tid : SV_DispatchThreadID) {
    int index = int(tid.x);
    if (index >= pc.n) return;
    data[pc.dataBase + index] += prefix[pc.prefixBase + (index / 128)];
}
