[shader("compute")]
[numthreads(1024, 1, 1)]
void main(
    uint tid: SV_DispatchThreadID,
    StructuredBuffer<float> a,
    StructuredBuffer<float> b,
    RWStructuredBuffer<float> c
)
{
    c[tid] = a[tid] + b[tid];
}
