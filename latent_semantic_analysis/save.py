import shutil
from typing import Any, Dict

import joblib
from sklearn.pipeline import Pipeline


def save_model(
    pipe: Pipeline,
    config: Dict[str, Any],
) -> None:
    """
    Save:
        - model pipeline (tf-idf + svd)
        - config
    :param Pipeline pipe: model pipeline (tf-idf + svd).
    :param Dict[str, Any] config: config.
    """

    # save pipe
    joblib.dump(pipe, config["path_to_save_model"])

    # save config
    shutil.copy2(config["path_to_config"], config["path_to_save_folder"])
