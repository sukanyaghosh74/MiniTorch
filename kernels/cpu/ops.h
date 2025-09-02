#pragma once
#include <cstdint>

void add_cpu(const float* a, const float* b, float* out, int64_t size);
void mul_cpu(const float* a, const float* b, float* out, int64_t size);
void matmul_cpu(const float* a, const float* b, float* out, int64_t m, int64_t k, int64_t n);
void conv2d_forward_cpu(/* args */);
void conv2d_backward_cpu(/* args */);
