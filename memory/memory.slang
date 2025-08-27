import sgl.device.print;

[shader("compute")]
[numthreads(10, 1, 1)]
void main(
    uint tid: SV_DispatchThreadID,
    StructuredBuffer<float> a
)
{
    if( tid == 0) {
        uint a_size = a.getCount();
        for(uint i = 0; i < a_size; i++)
            print("array[{}]={}", i, a[i]);
    }
}
