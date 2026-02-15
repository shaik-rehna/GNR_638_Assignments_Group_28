#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<int> shape;

    bool requires_grad;

    std::vector<std::shared_ptr<Tensor>> parents;
    std::function<void()> backward_fn;

    Tensor(const std::vector<float>& data,
           const std::vector<int>& shape,
           bool requires_grad=false);

    int numel() const;
    void zero_grad();
    void backward();

private:
    void build_topo(std::shared_ptr<Tensor> node,
                    std::vector<std::shared_ptr<Tensor>>& topo,
                    std::unordered_set<Tensor*>& visited);
};