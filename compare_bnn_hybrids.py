#!/usr/bin/env python3
"""Run both research deliverables and write one comparison CSV."""

from __future__ import annotations

import argparse

from hybrid_bnn_common import (
    BayesianNNPredictor,
    BranchFeatureEncoder,
    GsharePredictor,
    HybridBNNSimulator,
    TwoBitCounterPredictor,
    result_to_dict,
    write_csv_row,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare 2-bit + BNN against gshare + BNN.")
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-branches", type=int, default=None)
    parser.add_argument("--two-bit-pc-bits", type=int, default=12)
    parser.add_argument("--gshare-pc-bits", type=int, default=14)
    parser.add_argument("--gshare-history-bits", type=int, default=10)
    parser.add_argument("--gshare-counter-bits", type=int, default=2)
    parser.add_argument("--bnn-pc-bits", type=int, default=12)
    parser.add_argument("--bnn-history-bits", type=int, default=24)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--train-interval", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=512)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--weak-states", type=int, default=1)
    parser.add_argument("--energy-base", type=float, default=1.0)
    parser.add_argument("--energy-bnn-infer", type=float, default=8.0)
    parser.add_argument("--energy-bnn-train", type=float, default=2.0)
    parser.add_argument("--misprediction-penalty", type=int, default=12)
    return parser


def build_bnn(args: argparse.Namespace) -> tuple[BranchFeatureEncoder, BayesianNNPredictor]:
    encoder = BranchFeatureEncoder(args.bnn_pc_bits, args.bnn_history_bits)
    bnn = BayesianNNPredictor(
        input_size=encoder.input_size,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout_p=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        train_interval=args.train_interval,
        warmup=args.warmup,
        mc_samples=args.mc_samples,
    )
    return encoder, bnn


def main() -> None:
    args = build_parser().parse_args()
    for trace in args.trace:
        encoder, bnn = build_bnn(args)
        two_bit = HybridBNNSimulator(
            model_name="two_bit_bnn",
            base_predictor=TwoBitCounterPredictor(args.two_bit_pc_bits),
            feature_encoder=encoder,
            bnn_predictor=bnn,
            weak_states=args.weak_states,
            energy_base=args.energy_base,
            energy_bnn_infer=args.energy_bnn_infer,
            energy_bnn_train=args.energy_bnn_train,
            misprediction_penalty=args.misprediction_penalty,
        )
        result = two_bit.run(trace, max_branches=args.max_branches)
        write_csv_row(args.output_csv, result_to_dict(result))
        print(
            f"{result.model_name:>12}  {trace.split('/')[-1]:>14}  "
            f"acc={result.accuracy:.4f}  miss={result.misprediction_rate:.4f}  "
            f"energy={result.estimated_energy_units:.1f}"
        )

        encoder, bnn = build_bnn(args)
        gshare = HybridBNNSimulator(
            model_name="gshare_bnn",
            base_predictor=GsharePredictor(
                pc_bits=args.gshare_pc_bits,
                history_bits=args.gshare_history_bits,
                counter_bits=args.gshare_counter_bits,
            ),
            feature_encoder=encoder,
            bnn_predictor=bnn,
            weak_states=args.weak_states,
            energy_base=args.energy_base,
            energy_bnn_infer=args.energy_bnn_infer,
            energy_bnn_train=args.energy_bnn_train,
            misprediction_penalty=args.misprediction_penalty,
        )
        result = gshare.run(trace, max_branches=args.max_branches)
        write_csv_row(args.output_csv, result_to_dict(result))
        print(
            f"{result.model_name:>12}  {trace.split('/')[-1]:>14}  "
            f"acc={result.accuracy:.4f}  miss={result.misprediction_rate:.4f}  "
            f"energy={result.estimated_energy_units:.1f}"
        )


if __name__ == "__main__":
    main()
