#include <stdint.h>
extern "C" {
float bf16_to_float(uint16_t x) {
    uint32_t y = (uint32_t)x << 16;
    return *reinterpret_cast<float*>(&y);
}
void zero_f32(float* c) {
    for(int i=0; i<64*32; i++) c[i] = 0.0f;
}
void matmul_bf16_f32(uint16_t* a, uint16_t* b, float* c) {
    for(int i=0; i<64; i++) {
        for(int j=0; j<32; j++) {
            float sum = 0.0f;
            for(int k=0; k<64; k++) {
                sum += bf16_to_float(a[i*64+k]) * bf16_to_float(b[j*64+k]);
            }
            c[i*32+j] += sum;
        }
    }
}
}
