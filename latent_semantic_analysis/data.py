from typing import Any, Dict

import pandas as pd


def load_data(config: Dict[str, Any]) -> pd.Series:
    """
    Load data.
    :param Dict[str, Any] config: config.
    :return: Data.
    :rtype: pd.Series
    """

    text_column = config["data"]["text_column"]

    sep = config["data"]["sep"]

    data = pd.read_csv(
        config["data"]["data_path"],
        sep=sep,
        usecols=[text_column],
    )[text_column]

    return data
