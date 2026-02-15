#pragma once
#include <memory>
#include "tensor.h"

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a,
                            std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a,
                            std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a,
                               std::shared_ptr<Tensor> b);

                               