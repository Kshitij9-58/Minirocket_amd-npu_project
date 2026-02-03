#include <stdint.h>

extern "C" { // <--- THIS IS REQUIRED FOR C++

// Inputs: 128 floats (Time Series), 84 floats (Biases)
// Output: 84 floats (Features)
void minirocket_kernel(float* in_ts, float* in_bias, float* out_feat) {
    for (int k = 0; k < 84; k++) {
        float bias = in_bias[k];
        float sum = 0.0f;
        for (int t = 0; t < 128; t++) {
            if (in_ts[t] > bias) {
                sum += 1.0f;
            }
        }
        out_feat[k] = sum / 128.0f; 
    }
}

} // extern "C"