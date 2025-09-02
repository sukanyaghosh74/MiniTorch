#pragma once
#include <vector>
#include <string>
#include <memory>

namespace minitorch {

class Tensor {
public:
    Tensor(const std::vector<int64_t>& shape, const std::string& dtype, const std::string& device);
    ~Tensor();
    void* data();
    std::vector<int64_t> shape() const;
    std::string dtype() const;
    std::string device() const;
    Tensor add(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    Tensor sum() const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace minitorch
