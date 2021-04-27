import os
import argparse

import tensorflow as tf

from ragn import training


def weights(s):
    try:
        l = list(map(float, s.split(",")))
    except:
        raise argparse.ArgumentTypeError(
            "Class weights must be a sequence of two floats splited by commas"
        )

    if len(l) != 2:
        raise argparse.ArgumentTypeError(
            "Class weights must be a sequence of two floats splited by commas"
        )
    return tf.constant(l, dtype=tf.float32)


if __name__ == "__main__":
    root_dir = os.path.join(os.path.expanduser("~"), "FintelligenceData")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tr-size",
        type=int,
        default=500000,
        help="Size of training dataset",
    )
    p.add_argument(
        "--tr-path",
        type=str,
        default="training/All",
        help="Path to the trainig dataset",
    )
    p.add_argument(
        "--val-path",
        type=str,
        default="validation/All",
        help="Path to the validation dataset",
    )
    p.add_argument(
        "--log-path",
        type=str,
        default="logs",
        help="Path to save the assets",
    )
    p.add_argument(
        "--n-msg",
        type=int,
        default=40,
        help="Number of messages",
    )
    p.add_argument(
        "--n-epoch",
        type=int,
        default=10,
        help="Number of epochs",
    )
    p.add_argument(
        "--n-batch",
        type=int,
        default=64,
        help="Batch size",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Start the debug running, the functions are not compiled",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed",
    )
    p.add_argument(
        "--init-lr",
        type=float,
        default=5e-3,
        help="Initial learning rate",
    )
    p.add_argument(
        "--end-lr",
        type=float,
        default=1e-5,
        help="Final learning rate",
    )
    p.add_argument(
        "--decay-steps",
        type=int,
        default=100000,
        help="Number of steps to reach the final learning rate",
    )
    p.add_argument(
        "--power",
        type=int,
        default=3,
        help="The power of the polynomial decay related for learning rate",
    )
    p.add_argument(
        "--delta-time-to-validate",
        type=int,
        default=30,
        help="The interval time that the validation dataseet should be assessed",
    )
    p.add_argument(
        "--class-weight",
        type=weights,
        default=tf.constant([0.2, 1.0], dtype=tf.float32),
        help="The weight for each class (non-routing link and routing link)",
    )
    args = p.parse_args()
    training.train_ragn(**vars(args))
    # run(**vars(args))
