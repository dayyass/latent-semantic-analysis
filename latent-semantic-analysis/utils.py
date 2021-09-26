from argparse import ArgumentParser
import random
import numpy as np


def get_argparse() -> ArgumentParser:
    """
    Get argument parser.
    :return: argument parser.
    :rtype: ArgumentParser
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        help="Path to config",
    )

    return parser


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.
    :param int seed: seed.
    """

    random.seed(seed)
    np.random.seed(seed)
