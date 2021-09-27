import os
import unittest

from data.load_20newsgroups import load_20newsgroups
from latent_semantic_analysis.__main__ import train
from latent_semantic_analysis.config import load_default_config


class TestUsage(unittest.TestCase):
    """
    Class for testing pipeline.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        SetUp tests with config and data.
        """

        path_to_config = "config.yaml"

        if os.path.exists(path_to_config):
            os.remove(path_to_config)

        load_default_config()
        load_20newsgroups()

    def test_train(self) -> None:
        """
        Testing train function.
        """

        train(path_to_config="config.yaml")


if __name__ == "__main__":
    unittest.main()
