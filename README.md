# Hybrid BNN Branch Predictor

This folder is now focused only on the current assignment:
- [two_bit_bnn.py](/Users/abramtadros/Desktop/Lecture%20Slides/CDA%205106/BNN/two_bit_bnn.py)
- [gshare_bnn.py](/Users/abramtadros/Desktop/Lecture%20Slides/CDA%205106/BNN/gshare_bnn.py)
- [compare_bnn_hybrids.py](/Users/abramtadros/Desktop/Lecture%20Slides/CDA%205106/BNN/compare_bnn_hybrids.py)
- [hybrid_bnn_common.py](/Users/abramtadros/Desktop/Lecture%20Slides/CDA%205106/BNN/hybrid_bnn_common.py)

`main(1).cc` was only used as background for predictor structure and trace handling.
There are no MP2-specific scripts left in this folder.

## Models

`two_bit_bnn.py`
- baseline: 2-bit saturating counter
- fallback model: Bayesian-style neural network using MC dropout

`gshare_bnn.py`
- baseline: gshare predictor
- fallback model: Bayesian-style neural network using MC dropout

Both scripts:
- read the same trace format you already have
- invoke the BNN only when the baseline is uncertain
- train the BNN online while processing the trace
- report accuracy and normalized energy

## Run one model

```bash
python3 /Users/abramtadros/Desktop/Lecture\ Slides/CDA\ 5106/BNN/two_bit_bnn.py \
  --trace "/Users/abramtadros/Desktop/Lecture Slides/CDA 5106/MachineProblem2/BranchPrediction/traces/gcc_trace.txt" \
  --max-branches 100000 \
  --csv "/Users/abramtadros/Desktop/Lecture Slides/CDA 5106/BNN/results/two_bit_bnn.csv"
```

```bash
python3 /Users/abramtadros/Desktop/Lecture\ Slides/CDA\ 5106/BNN/gshare_bnn.py \
  --trace "/Users/abramtadros/Desktop/Lecture Slides/CDA 5106/MachineProblem2/BranchPrediction/traces/gcc_trace.txt" \
  --max-branches 100000 \
  --csv "/Users/abramtadros/Desktop/Lecture Slides/CDA 5106/BNN/results/gshare_bnn.csv"
```

## Compare both models

```bash
python3 /Users/abramtadros/Desktop/Lecture\ Slides/CDA\ 5106/BNN/compare_bnn_hybrids.py \
  --trace "/Users/abramtadros/Desktop/Lecture Slides/CDA 5106/MachineProblem2/BranchPrediction/traces/gcc_trace.txt" \
  --trace "/Users/abramtadros/Desktop/Lecture Slides/CDA 5106/MachineProblem2/BranchPrediction/traces/perl_trace.txt" \
  --trace "/Users/abramtadros/Desktop/Lecture Slides/CDA 5106/MachineProblem2/BranchPrediction/traces/jpeg_trace.txt" \
  --output-csv "/Users/abramtadros/Desktop/Lecture Slides/CDA 5106/BNN/results/bnn_hybrid_comparison.csv" \
  --max-branches 100000
```

## Online workflow

Each run is split into two phases:

1. Prediction phase
   For each branch, the baseline predicts first.
   If the baseline is weak, the BNN is invoked.

2. Learning phase
   After the true branch outcome is known, the BNN stores that example and periodically updates its weights during the same run.

Default online-training settings:
- `buffer_size = 4096`
- `warmup = 512`
- `train_interval = 16`
- `batch_size = 64`

This means the model learns as it sees more of the same trace.

## Compare these metrics

- `accuracy`
- `misprediction_rate`
- `bnn_invocation_rate`
- `estimated_energy_units`
- `estimated_penalty_cycles`

`estimated_energy_units` is a normalized comparison metric:
- each baseline prediction costs `energy_base`
- each BNN inference costs `energy_bnn_infer`
- each BNN training update costs `energy_bnn_train`

It is useful for comparing the two designs on the same traces, but it is not a real hardware pJ measurement.

## Important limitation

The provided traces only contain:
- branch PC
- actual outcome

So this project can compare `2-bit + BNN` against `gshare + BNN` directly, but it is not a full reproduction of the IEEE paper's exact ChampSim setup.
