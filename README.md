[![tests](https://github.com/dayyass/latent-semantic-analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/dayyass/latent-semantic-analysis/actions/workflows/tests.yml)
[![linter](https://github.com/dayyass/latent-semantic-analysis/actions/workflows/linter.yml/badge.svg)](https://github.com/dayyass/latent-semantic-analysis/actions/workflows/linter.yml)
[![codecov](https://codecov.io/gh/dayyass/latent-semantic-analysis/branch/main/graph/badge.svg?token=Y39Q5786DL)](https://codecov.io/gh/dayyass/latent-semantic-analysis)

[![python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://github.com/dayyass/latent-semantic-analysis#requirements)
[![release (latest by date)](https://img.shields.io/github/v/release/dayyass/latent-semantic-analysis)](https://github.com/dayyass/latent-semantic-analysis/releases/latest)
[![license](https://img.shields.io/github/license/dayyass/latent-semantic-analysis?color=blue)](https://github.com/dayyass/latent-semantic-analysis/blob/main/LICENSE)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-black)](https://github.com/dayyass/latent-semantic-analysis/blob/main/.pre-commit-config.yaml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![pypi version](https://img.shields.io/pypi/v/latent-semantic-analysis)](https://pypi.org/project/latent-semantic-analysis)
[![pypi downloads](https://img.shields.io/pypi/dm/latent-semantic-analysis)](https://pypi.org/project/latent-semantic-analysis)

### Latent Semantic Analysis
Pipeline for training **LSA** models using Scikit-Learn.

### Usage
Instead of writing custom code for latent semantic analysis, you just need:
1. install pipeline:
```shell script
pip install latent-semantic-analysis
```
2. run pipeline:
- either in **terminal**:
```shell script
lsa-train --path_to_config config.yaml
```
- or in **python**:
```python3
import latent_semantic_analysis

latent_semantic_analysis.train(path_to_config="config.yaml")
```

**NOTE**: more about config file [here](https://github.com/dayyass/latent-semantic-analysis/tree/main#config).

No data preparation is needed, only a **csv** file with raw text column (with arbitrary name).

#### Config
The user interface consists of only one files:
- [**config.yaml**](https://github.com/dayyass/latent-semantic-analysis/blob/main/config.yaml) - general configuration with sklearn **TF-IDF** and **SVD** parameters

Change **config.yaml** to create the desired configuration and train LSA model with the following command:
- **terminal**:
```shell script
lsa-train --path_to_config config.yaml
```
- **python**:
```python3
import latent_semantic_analysis

latent_semantic_analysis.train(path_to_config="config.yaml")
```

Default **config.yaml**:
```yaml
seed: 42
path_to_save_folder: models

# data
data:
  data_path: data/data.csv
  sep: ','
  text_column: text

# tf-idf
tf-idf:
  lowercase: true
  ngram_range: (1, 1)
  max_df: 1.0
  min_df: 1

# svd
svd:
  n_components: 10
  algorithm: arpack
```

**NOTE**: `tf-idf` and `svd` are sklearn [**TfidfVectorizer**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html?highlight=tfidf#sklearn.feature_extraction.text.TfidfVectorizer) and [**TruncatedSVD**](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) parameters correspondingly, so you can parameterize instances of these classes however you want.

#### Output
After training the model, the pipeline will return the following files:
- `model.joblib` - sklearn pipeline with LSA (TF-IDF and SVD steps)
- `config.yaml` - config that was used to train the model
- `logging.txt` - logging file
- `doc2topic.json` - document embeddings
- `term2topic.json` - term embeddings

### Requirements
Python >= 3.6
