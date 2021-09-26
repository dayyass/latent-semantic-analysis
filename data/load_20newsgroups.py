import os

import pandas as pd
from sklearn.datasets import fetch_20newsgroups


def load_20newsgroups() -> None:
    """
    Load 20newsgroups dataset.
    """

    bunch = fetch_20newsgroups(subset="all")

    df = pd.DataFrame(
        {
            "text": bunch.data,
            "target": bunch.target,
        }
    )
    df["target_name"] = df["target"].map(lambda x: bunch.target_names[x])
    df["target_name_short"] = df["target_name"].map(lambda x: x.split(".")[0])

    os.makedirs("data", exist_ok=True)

    df.to_csv("data/data.csv", index=False)


if __name__ == "__main__":
    load_20newsgroups()
