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

    # doc2topic from matrix U
    U = pipe.transform(data)
    doc2topic = {doc: U[idx].tolist() for idx, doc in enumerate(data)}

    # term2topic from matrix V
    V = pipe["svd"].components_.T
    term2topic = {
        term: V[idx].tolist() for term, idx in pipe["tf-idf"].vocabulary_.items()
    }

    logger.info("Done!")
    logger.info(f"TF-IDF number of features: {len(term2topic)}")

    # save model
    logger.info("Saving the model...")

    save_model(
        pipe=pipe,
        config=config,
        doc2topic=doc2topic,
        term2topic=term2topic,
    )

    logger.info("Done!")

    close_logger(logger)
