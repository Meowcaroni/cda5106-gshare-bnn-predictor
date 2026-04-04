# Team Project Summary: `2-bit + BNN` vs `gshare + BNN`

## Project goal

Our team project compares two hybrid branch predictors:

1. `2-bit + BNN`
2. `gshare + BNN`

The goal is to measure:
- prediction accuracy
- misprediction rate
- approximate energy usage

using the same branch trace inputs.

## Current code files

- `two_bit_bnn.py`
  Runs the hybrid predictor that uses a 2-bit saturating counter as the baseline and a Bayesian-style neural network as the fallback predictor.

- `gshare_bnn.py`
  Runs the hybrid predictor that uses gshare as the baseline and the same Bayesian-style neural network as the fallback predictor.

- `compare_bnn_hybrids.py`
  Runs both models on one or more traces and writes comparable results to one CSV.

- `hybrid_bnn_common.py`
  Shared implementation for:
  - trace parsing
  - 2-bit predictor
  - gshare predictor
  - feature encoding
  - Bayesian-style neural network
  - hybrid simulation loop

## High-level design

Both models use the same hybrid strategy:

1. The baseline predictor makes the first prediction.
2. If the baseline is confident, we use that prediction directly.
3. If the baseline is uncertain, we invoke the BNN.
4. After the actual branch outcome is known, we update:
   - the baseline predictor
   - the BNN training buffer
   - the BNN weights, occasionally

So the BNN is not used for every branch. It is meant to help on harder cases while the cheaper baseline handles easier cases.

## What the BNN actually is

The current implementation is not a fully formal Bayesian neural network in the strict research sense.

It is a Bayesian-style neural network using MC dropout:

- dropout is used during training
- dropout is also kept active during inference
- the same input is passed through the network multiple times
- the outputs are averaged
- the variation across those outputs acts like an uncertainty estimate

This is a common practical approximation for Bayesian behavior.

## BNN architecture

The network in `hybrid_bnn_common.py` is:

1. Linear layer
2. ReLU
3. Dropout
4. Linear layer
5. ReLU
6. Dropout
7. Final linear layer to one output

Default sizes:
- input size = `pc_feature_bits + history_bits`
- hidden layer 1 = `64`
- hidden layer 2 = `32`
- dropout = `0.25`

The final output is a single logit, which is converted to a probability with a sigmoid.

## What the BNN sees as input

For each branch, the BNN gets a feature vector made from:

1. Current branch PC bits
2. Recent branch outcome history

### PC features

The branch address is shifted right by 2 and converted into bit features.
Each bit is encoded as:
- `+1.0` if the bit is 1
- `-1.0` if the bit is 0

### History features

The recent branch outcomes are also encoded as:
- `+1.0` for taken
- `-1.0` for not-taken

So one training sample looks like:

- input:
  `[pc_bits..., recent_branch_history...]`
- label:
  `1.0` if the branch was taken, `0.0` if it was not taken

## How the BNN makes a prediction

For one branch:

1. Build the feature vector from the current PC and recent history.
2. Run the neural network multiple times with dropout still active.
3. Convert each output logit to a probability using sigmoid.
4. Average the probabilities.
5. If the mean probability is at least `0.5`, predict taken; otherwise predict not-taken.

Because dropout is active during inference, the outputs vary slightly across runs.
That variation gives a rough uncertainty estimate.

## How the BNN is trained

The current model is trained online from scratch while the trace is being processed.

There is no pre-trained model being loaded.

### Step-by-step training process

For one run:

1. Load the trace.
2. Process branches one by one.
3. For each branch:
   - build the feature vector
   - make a prediction
   - observe the true branch outcome
   - save the example for BNN training
   - occasionally update BNN weights during the same run

### Training target

The target is supervised classification:

- target = `1.0` if the branch was actually taken
- target = `0.0` if the branch was actually not taken

So the model is learning:

`(PC bits + recent branch history) -> taken / not taken`

### Batch training

The current defaults are:

- batch size = `64`
- replay-style buffer size = `4096`
- warmup = `512`
- train interval = every `16` examples

At each training step:

1. Randomly sample a minibatch from the buffer
2. Run them through the network
3. Compute binary classification loss using `BCEWithLogitsLoss`
4. Update weights using Adam optimizer

### Important note

This training is supervised classification, not reinforcement learning.

Even though the project and paper use the word "Bayesian" and discuss RL ideas, the current implementation is learning directly from correct branch labels, not from rewards or Q-values.

## How the 2-bit baseline works

The `2-bit + BNN` model uses a standard 2-bit saturating counter table:

- counter states: `0, 1, 2, 3`
- predict taken if counter >= `2`
- initialize counters to `2`

Uncertainty rule:
- only weak states trigger the BNN
- practically, that means the middle states are considered weak

So:
- strong not-taken and strong taken do not call the BNN
- weak not-taken and weak taken do call the BNN

## How the gshare baseline works

The `gshare + BNN` model uses:

- PC bits
- global branch history register
- XOR between PC index and history

It also uses saturating counters, with configurable size.

Defaults:
- baseline PC bits = `14`
- global history bits = `10`
- counter bits = `2`

Like the 2-bit version, if the counter is considered weak, the model calls the BNN.

## How energy is currently estimated

The code reports `estimated_energy_units`.

This is not real measured hardware power in pJ.
It is a normalized comparison metric.

The default energy model is:

- baseline prediction cost = `1.0`
- one BNN inference cost = `8.0`
- one BNN training step cost = `2.0`

This is useful for comparing the two designs on the same traces, but it should be described in the report as an abstract energy model, not a physical hardware measurement.

## How this compares to the research paper

The paper we used as inspiration describes a hybrid system that combines:

- a conventional 2-bit predictor
- a Bayesian Neural Network based Deep Q-Network (BNN-DQN)

The paper says:

- the simple baseline predicts easy branches
- the BNN-DQN is used for uncertain branches
- the model uses Bayesian uncertainty ideas
- the neural component is trained online
- the experiments are discussed in a Python/ChampSim context

### Similarities to the paper

Our current implementation matches the paper in these broad ideas:

1. Hybrid structure
   A simple branch predictor handles easy cases and the neural model is only used when needed.

2. Bayesian-style uncertainty
   We use MC dropout to approximate Bayesian predictive uncertainty.

3. Online learning
   The model trains while reading the trace.

4. Branch-history-based features
   The network uses branch-relevant information rather than unrelated program features.

### Differences from the paper

Our current implementation is simpler than the paper's described method.

1. We are not using DQN / Q-learning
   The paper describes a BNN-DQN / reinforcement-learning framing.
   Our code uses supervised binary classification instead.

2. We are not using rewards, actions, or Q-values
   The paper's RL framing treats prediction as a decision process.
   Our code directly learns from the true branch outcome label.

3. We do not use a target network
   DQN normally uses a separate target network for stability.
   Our code does not.

4. We do not implement Bellman updates
   That is a core RL piece in DQN. Our model does not do this.

5. We are not integrated with ChampSim
   Our current project works directly from the text trace files you already have.

6. Our "Bayesian" method is approximate
   We use MC dropout, not a heavier fully Bayesian parameter treatment.

### Best way to describe our model honestly

In the report, the safest wording is:

"Our implementation is a Bayesian-style hybrid neural predictor using Monte Carlo dropout, inspired by the BNN-based hybrid framework described in the reference paper, but simplified into an online supervised learning formulation for compatibility with the available branch traces."

That wording is accurate and defensible.

## Strengths of the current implementation

- Easy to run on a Mac CPU
- No CUDA required
- Uses your real course traces directly
- Clean comparison between `2-bit + BNN` and `gshare + BNN`
- Bayesian-style uncertainty is present through MC dropout
- Online training behavior is easy to explain

## Limitations of the current implementation

- No true RL / DQN logic
- No saved checkpoints yet
- No explicit train/validation/test split
- Energy model is normalized, not physical
- Results can vary by trace and parameter choices
- `gshare + BNN` is not guaranteed to beat `2-bit + BNN` on every trace

## Recommended next steps

If the team wants a stronger final version, the best improvements are:

1. Add checkpoint save/load for the BNN
2. Split traces into train/validation/test segments
3. Tune:
   - gshare PC bits
   - history length
   - counter width
   - dropout
   - weak-state threshold
4. Make the gate smarter so gshare only calls the BNN on truly difficult branches
5. If desired, move closer to the paper by replacing supervised learning with a more RL-like formulation

## One-sentence takeaway

The current project implements a practical CPU-friendly hybrid branch predictor where a simple baseline handles easy branches and a small Bayesian-style neural network learns from the trace online to help with harder branches, while remaining simpler than the paper's full BNN-DQN approach.
