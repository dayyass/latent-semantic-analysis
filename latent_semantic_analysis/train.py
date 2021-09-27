import logging
from typing import Any, Dict

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .data import load_data
from .save import save_model
from .utils import close_logger, set_seed


def _train(
    config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Function to train LSA model.
    :param Dict[str, Any] config: config.
    :param logging.Logger logger: logger.
    """

    # log config
    with open(config["path_to_config"], mode="r") as fp:
        logger.info(f"Config:\n\n{fp.read()}")

    # reproducibility
    set_seed(config["seed"])

    # load data
    logger.info("Loading data...")

    data = load_data(config)

    logger.info(f"Dataset size: {data.shape[0]}")

    # tf-idf
    vectorizer = TfidfVectorizer(**config["tf-idf"])

    # svd
    svd = TruncatedSVD(
        **config["svd"],
        random_state=config["seed"],
    )

    # pipeline
    pipe = Pipeline(
        [
            ("tf-idf", vectorizer),
            ("svd", svd),
        ],
        verbose=True,  # hardcoded
    )

    logger.info("Fitting LSA model...")

    pipe.fit(data)

    logger.info("Done!")
    logger.info(f"TF-IDF number of features: {len(pipe['tf-idf'].vocabulary_)}")

    # # metrics
    # logger.info("Calculating metrics...")

    # y_pred_train = pipe.predict(X_train)
    # classification_report_train = classification_report(
    #     y_true=y_train,
    #     y_pred=y_pred_train,
    #     target_names=target_names,
    # )

    # logger.info(f"Train classification report:\n\n{classification_report_train}")

    # y_pred_valid = pipe.predict(X_valid)
    # classification_report_valid = classification_report(
    #     y_true=y_valid,
    #     y_pred=y_pred_valid,
    #     target_names=target_names,
    # )

    # logger.info(f"Valid classification report:\n\n{classification_report_valid}")

    # save model
    logger.info("Saving the model...")

    save_model(
        pipe=pipe,
        config=config,
    )

    logger.info("Done!")

    close_logger(logger)