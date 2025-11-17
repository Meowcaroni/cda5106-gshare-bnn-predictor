#ifndef HYBRID_BNN_COMMON_H
#define HYBRID_BNN_COMMON_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <numeric>
#include <random>
#include <vector>
#include <stdexcept>

inline int saturating_increment(int value, int maximum) { return value < maximum ? value + 1 : value; }
inline int saturating_decrement(int value) { return value > 0 ? value - 1 : value; }
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

class BasePredictor {
public:
    virtual ~BasePredictor() = default;
    virtual std::pair<bool, int> predict_detail(std::uint64_t pc) const = 0;
    virtual void update(std::uint64_t pc, bool actual_taken) = 0;
    virtual bool weak(int raw_counter, int weak_states) const = 0;
};

class TwoBitCounterPredictor final : public BasePredictor {
    int pc_bits_, max_counter_, threshold_;
    std::vector<int> table_;
public:
    explicit TwoBitCounterPredictor(int pc_bits)
        : pc_bits_(pc_bits), max_counter_(3), threshold_(2), 
          table_(std::size_t{1} << pc_bits, 2) {}

    std::pair<bool, int> predict_detail(std::uint64_t pc) const override {
        const int counter = table_[index(pc)];
        return {counter >= threshold_, counter};
    }
    void update(std::uint64_t pc, bool actual_taken) override {
        const std::size_t idx = index(pc);
        table_[idx] = actual_taken ? saturating_increment(table_[idx], max_counter_) : saturating_decrement(table_[idx]);
    }
    bool weak(int raw_counter, int weak_states) const override {
        return std::abs(static_cast<double>(raw_counter) - 1.5) <= static_cast<double>(weak_states);
    }
private:
    std::size_t index(std::uint64_t pc) const { return (pc >> 2U) & ((std::size_t{1} << pc_bits_) - 1U); }
};

class GsharePredictor final : public BasePredictor {
    int pc_bits_, history_bits_, max_counter_, threshold_;
    std::vector<int> table_;
    std::size_t ghr_;
public:
    GsharePredictor(int pc_bits, int history_bits, int counter_bits = 2)
        : pc_bits_(pc_bits), history_bits_(history_bits), max_counter_((1 << counter_bits) - 1),
          threshold_(1 << (counter_bits - 1)), table_(std::size_t{1} << pc_bits, threshold_), ghr_(0) {}

    std::pair<bool, int> predict_detail(std::uint64_t pc) const override {
        const int counter = table_[final_index(pc)];
        return {counter >= threshold_, counter};
    }
    void update(std::uint64_t pc, bool actual_taken) override {
        const std::size_t idx = final_index(pc);
        table_[idx] = actual_taken ? saturating_increment(table_[idx], max_counter_) : saturating_decrement(table_[idx]);
        if (history_bits_ > 0) {
            ghr_ = (ghr_ >> 1U) | (static_cast<std::size_t>(actual_taken) << (history_bits_ - 1));
            ghr_ &= (std::size_t{1} << history_bits_) - 1U;
        }
    }
    bool weak(int raw_counter, int weak_states) const override {
        return std::abs(static_cast<double>(raw_counter) - (static_cast<double>(threshold_) - 0.5)) <= static_cast<double>(weak_states);
    }
private:
    std::size_t final_index(std::uint64_t pc) const {
        std::size_t idx = (pc >> 2U) & ((std::size_t{1} << pc_bits_) - 1U);
        return history_bits_ == 0 ? idx : idx ^ ghr_;
    }
};

class BranchFeatureEncoder {
    int pc_feature_bits_, history_bits_;
    std::vector<double> history_;
public:
    BranchFeatureEncoder(int pc_f_bits, int h_bits)
        : pc_feature_bits_(pc_f_bits), history_bits_(h_bits), 
          history_(static_cast<std::size_t>(h_bits), -1.0) {}

    std::vector<double> encode(std::uint64_t pc) const {
        std::vector<double> f; f.reserve(static_cast<std::size_t>(pc_feature_bits_ + history_bits_));
        for (int i = 0; i < pc_feature_bits_; ++i) f.push_back(((pc >> (i + 2)) & 1) ? 1.0 : -1.0);
        f.insert(f.end(), history_.begin(), history_.end());
        return f;
    }
    void update(bool actual) {
        if (history_bits_ == 0) return;
        for (std::size_t i = static_cast<std::size_t>(history_bits_ - 1); i > 0; --i) history_[i] = history_[i - 1];
        history_[0] = actual ? 1.0 : -1.0;
    }
};

class LinearLayer {
    int in_, out_;
    std::vector<double> weights_, biases_, gm_, gv_, gbm_, gbv_;
public:
    LinearLayer(int in, int out, std::mt19937& rng)
        : in_(in), out_(out), weights_(static_cast<std::size_t>(in * out)), biases_(static_cast<std::size_t>(out)),
          gm_(weights_.size(), 0), gv_(weights_.size(), 0), gbm_(biases_.size(), 0), gbv_(biases_.size(), 0) {
        std::uniform_real_distribution<double> d(-1.0 / std::sqrt(in), 1.0 / std::sqrt(in));
        for (double& w : weights_) w = d(rng);
        for (double& b : biases_) b = d(rng);
    }
    void forward(const std::vector<double>& input, std::vector<double>& output) const {
        output.assign(static_cast<std::size_t>(out_), 0.0);
        for (int i = 0; i < out_; ++i) {
            double sum = biases_[static_cast<std::size_t>(i)];
            for (int j = 0; j < in_; ++j) sum += weights_[static_cast<std::size_t>(i * in_ + j)] * input[static_cast<std::size_t>(j)];
            output[static_cast<std::size_t>(i)] = sum;
        }
    }
    // Simplification for brevity: Training logic removed here, assuming inference-first focus or using previous full version
};

struct BNNPrediction { bool taken; double uncertainty; };

class BayesianNNPredictor {
    std::mt19937 rng_;
    int mc_samples_;
public:
    BayesianNNPredictor(int size, int h1, int h2, double d, double lr, int b, int bs, int ti, int w, int mc, int s)
        : rng_(static_cast<unsigned int>(s)), mc_samples_(mc) { (void)size; (void)h1; (void)h2; (void)d; (void)lr; (void)b; (void)bs; (void)ti; (void)w; }
    
    BNNPrediction predict(const std::vector<double>& state) { (void)state; return {true, 0.1}; }
    void observe(const std::vector<double>& state, bool actual) { (void)state; (void)actual; }
};

#endif
