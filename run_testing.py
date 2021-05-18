import os
import argparse

from ragn import testing


def field(s):
    try:
        l = s.split(",")
    except:
        raise argparse.ArgumentTypeError(
            "Fields must be a sequence of strings splited by commas"
        )
    return dict(node=l)


if __name__ == "__main__":
    root_dir = os.path.join(os.path.expanduser("~"), "FintelligenceData")
    p = argparse.ArgumentParser()
    p.add_argument(
        "restore_from",
        type=str,
        help="Path from log-path that will be used to restore the best model",
    )
    p.add_argument(
        "--test-path",
        type=str,
        default="test/All",
        help="Path to the test dataset",
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
        default=55,
        help="Number of messages",
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
    args = p.parse_args()
    testing.test_ragn(**vars(args))
