#ifndef TWO_BIT_BNN_H
#define TWO_BIT_BNN_H

#include "modules.h"
#include "hybrid_bnn_common.h"

class two_bit_bnn {
    int weak_states_;
    TwoBitCounterPredictor base_predictor_;
    BranchFeatureEncoder feature_encoder_;
    BayesianNNPredictor bnn_predictor_;

public:
    explicit two_bit_bnn(O3_CPU* cpu)
        : weak_states_(1),
          base_predictor_(12),
          feature_encoder_(12, 24),
          bnn_predictor_(36, 64, 32, 0.25, 1e-3, 64, 4096, 16, 512, 8, 7)
    { (void)cpu; }

    bool predict_branch(uint64_t ip, uint64_t predicted_target, uint8_t always_taken, uint8_t branch_type) {
        const std::vector<double> state = feature_encoder_.encode(ip);
        const auto [base_prediction, raw_counter] = base_predictor_.predict_detail(ip);
        
        if (base_predictor_.weak(raw_counter, weak_states_)) {
            return bnn_predictor_.predict(state).taken;
        }
        return base_prediction;
    }

    void last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
        base_predictor_.update(ip, taken != 0);
        bnn_predictor_.observe(feature_encoder_.encode(ip), taken != 0);
        feature_encoder_.update(taken != 0);
    }
};
#endif
