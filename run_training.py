import os
import argparse

from ragn import training


def field(s):
    try:
        l = s.split(",")
    except:
        raise argparse.ArgumentTypeError(
            "Fields must be a sequence of strings splited by commas"
        )
    return dict(node=l)


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
    return l


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
        default="train/All",
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
        "--restore-from",
        type=str,
        default=None,
        help="Path from log-path that will be used to restore the last checkpoint",
    )
    p.add_argument(
        "--n-msg",
        type=int,
        default=55,
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
        default=72,
        help="Batch size",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Start the debug running, the functions are not compiled",
    )
    p.add_argument(
        "--bidim-solution",
        action="store_true",
        help="To use the outputs in one-hot vector format",
    )
    p.add_argument(
        "--scale",
        action="store_true",
        help="Scale edge and node features",
    )
    p.add_argument(
        "--input-fields",
        type=field,
        default=None,
        help="Fields to be considered as node input features",
    )
    p.add_argument(
        "--opt",
        type=str,
        default="adam",
        help="Optimizer",
    )
    p.add_argument(
        "--sufix-name",
        type=str,
        default="",
        help="Sufix name for log dir",
    )
    p.add_argument(
        "--dropped-msg-ratio",
        type=float,
        default=0.35,
        help="Percentage of partial results from message passing that will not be consider in loss function",
    )
    p.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="Number of layers of each MLP",
    )
    p.add_argument(
        "--hidden-size",
        type=int,
        default=24,
        help="The base for the number of neurons in the hidden layers for both MLP and LSTM",
    )
    p.add_argument(
        "--rnn-depth",
        type=int,
        default=2,
        help="The number of LSTM that will be concatenated",
    )
    p.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="The number of heads used to make the link decision",
    )
    p.add_argument(
        "--n-att",
        type=int,
        default=3,
        help="Number of multihead attention will be used to make the link decision",
    )
    p.add_argument(
        "--create-offset",
        action="store_true",
        help="Create offset trainable parameter in the layer normalization",
    )
    p.add_argument(
        "--create-scale",
        action="store_true",
        help="Create scale trainable parameter in the layer normalization",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2,
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
        default=45000,
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
        default=[0.4, 1.0],
        help="The weight for each class (non-routing link and routing link)",
    )
    args = p.parse_args()
    training.train_ragn(**vars(args))
