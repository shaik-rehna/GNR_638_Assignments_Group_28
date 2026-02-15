#include "tensor.h"
#include <algorithm>

Tensor::Tensor(const std::vector<float>& data,
               const std::vector<int>& shape,
               bool requires_grad)
    : data(data), shape(shape), requires_grad(requires_grad)
{
    grad.resize(data.size(), 0.0f);
}

int Tensor::numel() const {
    return data.size();
}

void Tensor::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

void Tensor::build_topo(std::shared_ptr<Tensor> node,
                        std::vector<std::shared_ptr<Tensor>>& topo,
                        std::unordered_set<Tensor*>& visited) {

    if (visited.find(node.get()) == visited.end()) {
        visited.insert(node.get());

        for (auto& parent : node->parents) {
            build_topo(parent, topo, visited);
        }

        topo.push_back(node);
    }
}

void Tensor::backward() {

    if (!requires_grad)
        return;

    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;

    build_topo(shared_from_this(), topo, visited);

    // Initialize gradient of final node (dL/dL = 1)
    grad.assign(data.size(), 1.0f);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
        }
    }

    // CRITICAL: Free computational graph to prevent memory leak
    for (auto &node : topo) {
        node->parents.clear();
        node->backward_fn = nullptr;
    }
}
