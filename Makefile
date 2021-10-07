all:
	python -m latent_semantic_analysis --path_to_config config.yaml
clean:
	rm -rf models/model*
