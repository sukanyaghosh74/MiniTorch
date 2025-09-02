// CPU kernels for minitorch
#include "ops.h"

void add_cpu(const float* a, const float* b, float* out, int64_t size) {
    for (int64_t i = 0; i < size; ++i) out[i] = a[i] + b[i];
}

void mul_cpu(const float* a, const float* b, float* out, int64_t size) {
    for (int64_t i = 0; i < size; ++i) out[i] = a[i] * b[i];
}

void matmul_cpu(const float* a, const float* b, float* out, int64_t m, int64_t k, int64_t n) {
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int64_t l = 0; l < k; ++l)
                sum += a[i * k + l] * b[l * n + j];
            out[i * n + j] = sum;
        }
}

void conv2d_forward_cpu(/* args */) {
    // TODO: implement
}

void conv2d_backward_cpu(/* args */) {
    // TODO: implement
}
