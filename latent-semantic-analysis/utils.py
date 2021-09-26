from argparse import ArgumentParser


def get_argparse() -> ArgumentParser:
    """
    Get argument parser.
    :return: argument parser.
    :rtype: ArgumentParser
    """

    parser = ArgumentParser(prog="text-clf-train")
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        help="Path to config",
    )

    return parser
