#include "autograd.h"

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a,
                            std::shared_ptr<Tensor> b) {

    std::vector<float> result(a->numel());

    // Case 1: same shape
    if (a->numel() == b->numel()) {

        for (int i = 0; i < a->numel(); ++i) {
            result[i] = a->data[i] + b->data[i];
        }
    }
    // Case 2: broadcast bias [1, p] to [m, p]
    else if (b->shape[0] == 1 && a->shape.size() == 2) {

        int m = a->shape[0];
        int p = a->shape[1];

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                result[i * p + j] =
                    a->data[i * p + j] +
                    b->data[j];
            }
        }
    }
    else {
        throw std::runtime_error("Unsupported add broadcasting case");
    }

    auto out = std::make_shared<Tensor>(
        result,
        a->shape,
        a->requires_grad || b->requires_grad
    );

    out->parents = {a, b};

    out->backward_fn = [a, b, out]() {

        int total = out->numel();

        // dA
        if (a->requires_grad) {
            for (int i = 0; i < total; ++i)
                a->grad[i] += out->grad[i];
        }

        // dB (handle broadcast)
        if (b->requires_grad) {

            if (a->numel() == b->numel()) {

                for (int i = 0; i < total; ++i)
                    b->grad[i] += out->grad[i];
            }
            else {
                int m = a->shape[0];
                int p = a->shape[1];

                for (int j = 0; j < p; ++j) {
                    float grad_sum = 0.0f;
                    for (int i = 0; i < m; ++i) {
                        grad_sum += out->grad[i * p + j];
                    }
                    b->grad[j] += grad_sum;
                }
            }
        }

        // FREE GRAPH
        out->parents.clear();
        out->backward_fn = nullptr;
    };

    return out;
}

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a,
                            std::shared_ptr<Tensor> b) {

    std::vector<float> result(a->numel());

    for (int i = 0; i < a->numel(); ++i) {
        result[i] = a->data[i] * b->data[i];
    }

    auto out = std::make_shared<Tensor>(
        result,
        a->shape,
        a->requires_grad || b->requires_grad
    );

    out->parents = {a, b};

    out->backward_fn = [a, b, out]() {

        for (int i = 0; i < out->numel(); ++i) {

            if (a->requires_grad)
                a->grad[i] += b->data[i] * out->grad[i];

            if (b->requires_grad)
                b->grad[i] += a->data[i] * out->grad[i];
        }

        // FREE GRAPH
        out->parents.clear();
        out->backward_fn = nullptr;
    };

    return out;
}

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a) {

    float total = 0.0f;

    for (int i = 0; i < a->numel(); ++i) {
        total += a->data[i];
    }

    std::vector<float> result = { total };

    auto out = std::make_shared<Tensor>(
        result,
        std::vector<int>{1},
        a->requires_grad
    );

    out->parents = {a};

    out->backward_fn = [a, out]() {

        for (int i = 0; i < a->numel(); ++i) {

            if (a->requires_grad)
                a->grad[i] += out->grad[0];  // broadcast scalar gradient
        }

        // FREE GRAPH
        out->parents.clear();
        out->backward_fn = nullptr;
    };

    return out;
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a,
                               std::shared_ptr<Tensor> b) {

    int m = a->shape[0];
    int n = a->shape[1];
    int p = b->shape[1];

    std::vector<float> result(m * p, 0.0f);

    // Forward
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i * p + j] +=
                    a->data[i * n + k] *
                    b->data[k * p + j];
            }
        }
    }

    auto out = std::make_shared<Tensor>(
        result,
        std::vector<int>{m, p},
        a->requires_grad || b->requires_grad
    );

    out->parents = {a, b};

    out->backward_fn = [a, b, out, m, n, p]() {

        // dA = dC * B^T
        if (a->requires_grad) {
            for (int i = 0; i < m; ++i) {
                for (int k = 0; k < n; ++k) {
                    float grad_sum = 0.0f;
                    for (int j = 0; j < p; ++j) {
                        grad_sum +=
                            out->grad[i * p + j] *
                            b->data[k * p + j];
                    }
                    a->grad[i * n + k] += grad_sum;
                }
            }
        }

        // dB = A^T * dC
        if (b->requires_grad) {
            for (int k = 0; k < n; ++k) {
                for (int j = 0; j < p; ++j) {
                    float grad_sum = 0.0f;
                    for (int i = 0; i < m; ++i) {
                        grad_sum +=
                            a->data[i * n + k] *
                            out->grad[i * p + j];
                    }
                    b->grad[k * p + j] += grad_sum;
                }
            }
        }

        //  FREE GRAPH
        out->parents.clear();
        out->backward_fn = nullptr;
    };

    return out;
}