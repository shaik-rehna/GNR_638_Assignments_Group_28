#include "linear.h"
#include "../core/autograd.h"
#include <cstdlib>
#include <cmath>

// --------------------------------------------------
// Constructor (He Initialization)
// --------------------------------------------------
Linear::Linear(int in_features, int out_features)
    : in_features(in_features),
      out_features(out_features) {

    std::vector<float> w_data(in_features * out_features);

    // -------- He Initialization --------
    float fan_in = in_features;
    float scale = std::sqrt(2.0f / fan_in);

    for (auto &v : w_data) {
        float r = (float)std::rand() / RAND_MAX;  // [0,1]
        r = r * 2.0f - 1.0f;                      // [-1,1]
        v = r * scale;
    }
    // -----------------------------------

    weight = std::make_shared<Tensor>(
        w_data,
        std::vector<int>{in_features, out_features},
        true
    );

    std::vector<float> b_data(out_features, 0.0f);

    bias = std::make_shared<Tensor>(
        b_data,
        std::vector<int>{1, out_features},
        true
    );
}

// --------------------------------------------------
// Forward
// --------------------------------------------------
std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    auto out = matmul(input, weight);
    return add(out, bias);
}

// --------------------------------------------------
// Parameters 
// --------------------------------------------------
std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
    return {weight, bias};
}