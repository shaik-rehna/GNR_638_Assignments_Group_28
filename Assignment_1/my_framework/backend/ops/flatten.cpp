#include "flatten.h"

std::shared_ptr<Tensor> flatten(std::shared_ptr<Tensor> input) {

    int batch = input->shape[0];

    int features = 1;
    for (int i = 1; i < input->shape.size(); ++i)
        features *= input->shape[i];

    auto out = std::make_shared<Tensor>(
        input->data,
        std::vector<int>{batch, features},
        input->requires_grad
    );

    out->parents = {input};

    out->backward_fn = [input, out]() {

        if (input->requires_grad) {
            for (int i = 0; i < input->numel(); ++i)
                input->grad[i] += out->grad[i];
        }

        // FREE GRAPH
        out->parents.clear();
        out->backward_fn = nullptr;
    };

    return out;
}