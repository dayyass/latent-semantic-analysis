all:
	python -m latent_semantic_analysis --path_to_config config.yaml
coverage:
	coverage run -m unittest discover && coverage report -m
clean:
	rm -rf models/model*
