import os
import argparse

from ragn import training


def enc_conf(s):
    try:
        blocks = s.split(":")
    except:
        raise argparse.ArgumentTypeError("Field must has `:`")
    conf = []
    for b in blocks[:-1]:
        p = b.split(",")
        conf.append([[int(p[0]), int(p[1]), p[2]], [int(p[3]), int(p[4]), p[5]]])

    conf.append([])
    b = blocks[-1].split(",")
    for p in b:
        conf[-1].append(int(p))
    return conf


def mlp_conf(s):
    try:
        l = s.split(",")
    except:
        raise argparse.ArgumentTypeError("Field must has `,`")
    conf = []
    for p in l:
        conf.append(int(p))
    return conf


def rnn_conf(s):
    try:
        l = s.split(",")
    except:
        raise argparse.ArgumentTypeError("Field must has `,`")
    conf = [int(l[0]), int(l[1])]
    return conf


def decision_conf(s):
    try:
        l = s.split(",")
    except:
        raise argparse.ArgumentTypeError("Field must has `,`")
    conf = [int(l[0]), int(l[1])]
    return conf


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
        "--n-msg",
        type=int,
        default=20,
        help="Number of messages",
    )
    p.add_argument(
        "--n-epoch",
        type=int,
        default=60,
        help="Number of epochs",
    )
    p.add_argument(
        "--n-batch",
        type=int,
        default=128,
        help="Batch size",
    )
    p.add_argument(
        "--restore-from",
        type=str,
        default=None,
        help="Path from log-path that will be used to restore the last checkpoint",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Debugging mode, the functions are not compiled",
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
        default="rmsprop",
        help="Optimizer",
    )
    p.add_argument(
        "--dropped-msg-ratio",
        type=float,
        default=0.35,
        help="Percentage of partial results from message passing that will not be consider in loss function",
    )
    p.add_argument(
        "--enc-conf",
        type=enc_conf,
        default=[
            [[64, 8, 1, "SAME"], [8, 1, "SAME"]],
            [[32, 8, 1, "SAME"], [8, 8, "VALID"]],
            [128, 64, 32],
        ],
        help="Configuration of the encoder.",
    )
    p.add_argument(
        "--mlp-conf",
        type=mlp_conf,
        default=[64, 32, 32],
        help="Configuration of the MLP.",
    )
    p.add_argument(
        "--rnn-conf",
        type=rnn_conf,
        default=[32, 1],
        help="Configuration of the LSTM.",
    )

    p.add_argument(
        "--decision-conf",
        type=decision_conf,
        default=[2, 3],
        help="Configuration of the Transformer.",
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
        default=[0.4, 1.0],
        help="The weight for each class (non-routing link and routing link)",
    )
    args = p.parse_args()
    training.train_ragn(**vars(args))
