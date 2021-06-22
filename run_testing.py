import os
import argparse

from ragn import testing


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
        default=20,
        help="Number of messages",
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
        "--enc-conf",
        type=enc_conf,
        default=[
            # [[64, 8, 1, "SAME"], [8, 1, "SAME"]],
            [[32, 8, 1, "SAME"], [8, 8, "VALID"]],
            [64, 32, 32],
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
        default=[32, 2],
        help="Configuration of the LSTM.",
    )
    p.add_argument(
        "--decision-conf",
        type=decision_conf,
        default=[2, 4],
        help="Configuration of the Transformer.",
    )
    p.add_argument(
        "--create-offset",
        action="store_false",
        help="Create offset trainable parameter in the layer normalization",
    )
    p.add_argument(
        "--create-scale",
        action="store_false",
        help="Create scale trainable parameter in the layer normalization",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Start the debug running, the functions are not compiled",
    )
    args = p.parse_args()
    testing.test_ragn(**vars(args))
