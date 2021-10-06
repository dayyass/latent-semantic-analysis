import json
import shutil
from typing import Any, Dict, List

import joblib
from sklearn.pipeline import Pipeline


def save_model(
    pipe: Pipeline,
    config: Dict[str, Any],
    doc2topic: Dict[str, List[float]],
    term2topic: Dict[str, List[float]],
) -> None:
    """
    Save:
        - model pipeline (tf-idf + svd)
        - config
        - mapping document to topics embedding
        - mapping term to topics embedding
    :param Pipeline pipe: model pipeline (tf-idf + svd).
    :param Dict[str, Any] config: config.
    :param Dict[str, Any] doc2topic: mapping document to topics embedding.
    :param Dict[str, Any] term2topic: mapping term to topics embedding.
    """

    # save pipe
    joblib.dump(pipe, config["path_to_save_model"])

    # save config
    shutil.copy2(config["path_to_config"], config["path_to_save_folder"])

    # save mappings
    with open(config["path_to_save_model_doc2topic"], mode="w") as fp:
        json.dump(doc2topic, fp, ensure_ascii=False, indent=4)

    with open(config["path_to_save_model_term2topic"], mode="w") as fp:
        json.dump(term2topic, fp, ensure_ascii=False, indent=4)
