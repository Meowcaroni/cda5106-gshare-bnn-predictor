#!/usr/bin/env python3
"""
Shared utilities for the course-project research scripts:

- two_bit_bnn.py
- gshare_bnn.py

These scripts compare a low-cost baseline predictor against a Bayesian-style
neural model that is only invoked when the baseline is uncertain.

The Bayesian behavior is approximated with MC dropout:
- dropout remains active during inference
- multiple stochastic forward passes estimate predictive mean/uncertainty
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import deque
# Add [Suggestion 2]: from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
# Suggestion 2: If using Python 3.10+, all these aliases are deprecated and can be eliminated
# Remove [Suggestion 2]:
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple 

import torch
import torch.nn as nn


TraceEntry = Tuple[int, bool] # Replace [Suggestion 2]: TraceEntry = tuple[int, bool]


def parse_trace(trace_path: str, max_branches: Optional[int] = None) -> Iterator[TraceEntry]: # Replace [Suggestion 2]: max_branches: int | None = None
    with open(trace_path, "r", encoding="ascii") as trace_file:
        for index, line in enumerate(trace_file):
            if max_branches is not None and index >= max_branches:
                return
            if not line.strip():
                continue
            pc_text, outcome_text = line.split()
            yield int(pc_text, 16), outcome_text == "t"


def saturating_increment(value: int, maximum: int) -> int:
    return value + 1 if value < maximum else value


def saturating_decrement(value: int) -> int:
    return value - 1 if value > 0 else 0


class TwoBitCounterPredictor:
    def __init__(self, pc_bits: int):
        self.pc_bits = pc_bits
        self.max_counter = 3
        self.threshold = 2
        self.table = [2] * (1 << pc_bits)

    def index(self, pc: int) -> int:
        return (pc >> 2) & ((1 << self.pc_bits) - 1)

    def predict_detail(self, pc: int) -> Tuple[bool, int]: # Replace [Suggestion 2]: tuple[bool, int]
        counter = self.table[self.index(pc)]
        return counter >= self.threshold, counter

    def update(self, pc: int, actual_taken: bool) -> None:
        idx = self.index(pc)
        if actual_taken:
            self.table[idx] = saturating_increment(self.table[idx], self.max_counter)
        else:
            self.table[idx] = saturating_decrement(self.table[idx])

    def weak(self, raw_counter: int, weak_states: int = 1) -> bool:
        return abs(raw_counter - 1.5) <= weak_states

class GsharePredictor:
    def __init__(self, pc_bits: int, history_bits: int, counter_bits: int = 2):
        if history_bits > pc_bits:
            raise ValueError("history_bits must be <= pc_bits")
        self.pc_bits = pc_bits
        self.history_bits = history_bits
        self.counter_bits = counter_bits
        self.max_counter = (1 << counter_bits) - 1
        self.threshold = 1 << (counter_bits - 1)
        self.table = [self.threshold] * (1 << pc_bits)
        self.ghr = 0

    def pc_index(self, pc: int) -> int:
        return (pc >> 2) & ((1 << self.pc_bits) - 1)

    def final_index(self, pc: int) -> int:
        index = self.pc_index(pc)
        if self.history_bits == 0:
            return index
        return index ^ self.ghr

    def predict_detail(self, pc: int) -> Tuple[bool, int]: # Replace [Suggestion 2]: tuple[bool, int]
        counter = self.table[self.final_index(pc)]
        return counter >= self.threshold, counter

    def update(self, pc: int, actual_taken: bool) -> None:
        idx = self.final_index(pc)
        if actual_taken:
            self.table[idx] = saturating_increment(self.table[idx], self.max_counter)
        else:
            self.table[idx] = saturating_decrement(self.table[idx])
        self.update_history(actual_taken)

    def update_history(self, actual_taken: bool) -> None:
        if self.history_bits == 0:
            return
        # Suggestion: Potentially change GHR update procedure to shift left like in class example
        self.ghr >>= 1 # Replace [Suggestion 1]: "self.ghr <<= 1"
        if actual_taken:
            self.ghr |= 1 << (self.history_bits - 1) # Replace [Suggestion 1]: "self.ghr |= 1"
        self.ghr &= (1 << self.history_bits) - 1 # Mask away non-history bits

    def weak(self, raw_counter: int, weak_states: int = 1) -> bool:
        return abs(raw_counter - (self.threshold - 0.5)) <= weak_states


class BranchFeatureEncoder:
    """Encodes branch PC bits and recent history for the BNN."""

    def __init__(self, pc_feature_bits: int, history_bits: int):
        self.pc_feature_bits = pc_feature_bits
        self.history_bits = history_bits
        self.history: Deque[float] = deque([-1.0] * history_bits, maxlen=history_bits)

    @property
    def input_size(self) -> int:
        return self.pc_feature_bits + self.history_bits

    def encode(self, pc: int) -> List[float]: # Replace [Suggestion 2]: list[float]
        shifted_pc = pc >> 2
        pc_features = [1.0 if ((shifted_pc >> bit) & 1) else -1.0 for bit in range(self.pc_feature_bits)]
        return pc_features + list(self.history)

    def update(self, actual_taken: bool) -> None:
        if self.history_bits == 0:
            return
        self.history.appendleft(1.0 if actual_taken else -1.0)


class BayesianBranchNet(nn.Module):
    def __init__(self, input_size: int, hidden1: int = 64, hidden2: int = 32, dropout_p: float = 0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BayesianNNPredictor:
    """Small online-trained MC-dropout predictor."""

    def __init__(
        self,
        input_size: int,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout_p: float = 0.25,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 4096,
        train_interval: int = 16,
        warmup: int = 512,
        mc_samples: int = 8,
        seed: int = 7,
    ):
        random.seed(seed)
        torch.manual_seed(seed)
        self.model = BayesianBranchNet(input_size, hidden1=hidden1, hidden2=hidden2, dropout_p=dropout_p)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.warmup = warmup
        self.mc_samples = mc_samples
        self.examples_seen = 0
        self.training_steps = 0
        self.buffer: Deque[Tuple[List[float], float]] = deque(maxlen=buffer_size) # Replace [Suggestion 2]: self.buffer: deque[tuple[list[float], float]] = deque(maxlen=buffer_size)

    def predict(self, state: Sequence[float]) -> Tuple[bool, float, float]: # Replace [Suggestion 2]: tuple [bool, float, float]
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self.model.train()
        probs = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.model(x)
                probs.append(torch.sigmoid(logits).item())
        mean_prob = sum(probs) / len(probs)
        variance = sum((prob - mean_prob) ** 2 for prob in probs) / len(probs)
        uncertainty = math.sqrt(variance)
        return mean_prob >= 0.5, mean_prob, uncertainty

    def observe(self, state: Sequence[float], actual_taken: bool) -> int:
        label = 1.0 if actual_taken else 0.0
        self.buffer.append((list(state), label))
        self.examples_seen += 1

        steps_before = self.training_steps
        if len(self.buffer) >= self.batch_size and self.examples_seen >= self.warmup:
            if self.examples_seen % self.train_interval == 0:
                self._train_once()
        return self.training_steps - steps_before

    def _train_once(self) -> None:
        batch = random.sample(self.buffer, self.batch_size)
        features = torch.tensor([feature for feature, _ in batch], dtype=torch.float32)
        labels = torch.tensor([[label] for _, label in batch], dtype=torch.float32)
        self.model.train()
        logits = self.model(features)
        loss = self.loss_fn(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1


@dataclass
class HybridResult:
    model_name: str
    trace: str
    predictions: int
    mispredictions: int
    accuracy: float
    misprediction_rate: float
    bnn_invocations: int
    bnn_invocation_rate: float
    bnn_training_steps: int
    estimated_energy_units: float
    estimated_penalty_cycles: int
    base_correct: int
    bnn_used_correct: int


class HybridBNNSimulator:
    def __init__(
        self,
        model_name: str,
        base_predictor: object,
        feature_encoder: BranchFeatureEncoder,
        bnn_predictor: BayesianNNPredictor,
        weak_states: int = 1,
        # TODO: Adjust abstract constants below based on foundational research paper (2-bit/BNN hybrid) and other resources for justifiable realism
        energy_base: float = 1.0,
        energy_bnn_infer: float = 8.0,
        energy_bnn_train: float = 2.0,
        misprediction_penalty: int = 12,
    ):
        self.model_name = model_name
        self.base_predictor = base_predictor
        self.feature_encoder = feature_encoder
        self.bnn_predictor = bnn_predictor
        self.weak_states = weak_states
        self.energy_base = energy_base
        self.energy_bnn_infer = energy_bnn_infer
        self.energy_bnn_train = energy_bnn_train
        self.misprediction_penalty = misprediction_penalty

    def run(self, trace_path: str, max_branches: Optional[int] = None) -> HybridResult: # Replace [Suggestion 2]: max_branches: int | None = None
        total_predictions = 0
        mispredictions = 0
        base_correct = 0
        bnn_used_correct = 0
        bnn_invocations = 0
        training_steps = 0
        energy_units = 0.0

        for pc, actual_taken in parse_trace(trace_path, max_branches=max_branches):
            total_predictions += 1
            energy_units += self.energy_base

            state = self.feature_encoder.encode(pc)
            base_prediction, raw_counter = self.base_predictor.predict_detail(pc)
            if base_prediction == actual_taken:
                base_correct += 1

            use_bnn = self.base_predictor.weak(raw_counter, weak_states=self.weak_states)
            final_prediction = base_prediction
            if use_bnn:
                final_prediction, _, _ = self.bnn_predictor.predict(state)
                bnn_invocations += 1
                energy_units += self.energy_bnn_infer
                if final_prediction == actual_taken:
                    bnn_used_correct += 1

            if final_prediction != actual_taken:
                mispredictions += 1

            self.base_predictor.update(pc, actual_taken)
            added_training_steps = self.bnn_predictor.observe(state, actual_taken)
            if added_training_steps:
                training_steps += added_training_steps
                energy_units += added_training_steps * self.energy_bnn_train
            self.feature_encoder.update(actual_taken)

        accuracy = 0.0 if total_predictions == 0 else (total_predictions - mispredictions) / total_predictions
        misprediction_rate = 0.0 if total_predictions == 0 else mispredictions / total_predictions
        bnn_invocation_rate = 0.0 if total_predictions == 0 else bnn_invocations / total_predictions
        return HybridResult(
            model_name=self.model_name,
            trace=trace_path,
            predictions=total_predictions,
            mispredictions=mispredictions,
            accuracy=accuracy,
            misprediction_rate=misprediction_rate,
            bnn_invocations=bnn_invocations,
            bnn_invocation_rate=bnn_invocation_rate,
            bnn_training_steps=training_steps,
            estimated_energy_units=energy_units,
            estimated_penalty_cycles=mispredictions * self.misprediction_penalty,
            base_correct=base_correct,
            bnn_used_correct=bnn_used_correct,
        )


def result_to_dict(result: HybridResult) -> Dict[str, object]: # Replace [Suggestion 2]: -> dict[str, object]
    data = asdict(result)
    data["accuracy"] = round(result.accuracy, 6)
    data["misprediction_rate"] = round(result.misprediction_rate, 6)
    data["bnn_invocation_rate"] = round(result.bnn_invocation_rate, 6)
    data["estimated_energy_units"] = round(result.estimated_energy_units, 3)
    return data


def write_csv_row(output_path: str, row: Dict[str, object]) -> None:  # Replace [Suggestion 1]: row: dict[str, object]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with open(output, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_json_result(result: HybridResult, csv_path: Optional[str] = None) -> None: # Replace [Suggestion 2]: csv_path: str | None = None
    row = result_to_dict(result)
    if csv_path:
        write_csv_row(csv_path, row)
    print(json.dumps(row, indent=2, sort_keys=True))
