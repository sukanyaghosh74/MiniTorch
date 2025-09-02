#include "../include/tensor.h"
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cstring>

namespace minitorch {

struct Tensor::Impl {
    std::vector<int64_t> shape;
    std::string dtype;
    std::string device;
    std::vector<float> data;
};

Tensor::Tensor(const std::vector<int64_t>& shape, const std::string& dtype, const std::string& device) {
    impl_ = std::make_shared<Impl>();
    impl_->shape = shape;
    impl_->dtype = dtype;
    impl_->device = device;
    int64_t size = 1;
    for (auto d : shape) size *= d;
    impl_->data.resize(size, 0.0f);
}

Tensor::~Tensor() {}

void* Tensor::data() { return impl_->data.data(); }
std::vector<int64_t> Tensor::shape() const { return impl_->shape; }
std::string Tensor::dtype() const { return impl_->dtype; }
std::string Tensor::device() const { return impl_->device; }

Tensor Tensor::add(const Tensor& other) const {
    Tensor out(impl_->shape, impl_->dtype, impl_->device);
    for (size_t i = 0; i < impl_->data.size(); ++i) {
        out.impl_->data[i] = impl_->data[i] + other.impl_->data[i];
    }
    return out;
}

Tensor Tensor::mul(const Tensor& other) const {
    Tensor out(impl_->shape, impl_->dtype, impl_->device);
    for (size_t i = 0; i < impl_->data.size(); ++i) {
        out.impl_->data[i] = impl_->data[i] * other.impl_->data[i];
    }
    return out;
}

Tensor Tensor::matmul(const Tensor& other) const {
    // Only supports 2D for demo
    int64_t m = impl_->shape[0];
    int64_t k = impl_->shape[1];
    int64_t n = other.impl_->shape[1];
    Tensor out({m, n}, impl_->dtype, impl_->device);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int64_t l = 0; l < k; ++l) {
                sum += impl_->data[i * k + l] * other.impl_->data[l * n + j];
            }
            out.impl_->data[i * n + j] = sum;
        }
    }
    return out;
}

Tensor Tensor::sum() const {
    float s = std::accumulate(impl_->data.begin(), impl_->data.end(), 0.0f);
    Tensor out({1}, impl_->dtype, impl_->device);
    out.impl_->data[0] = s;
    return out;
}

} // namespace minitorch
