all:
	python -m latent_semantic_analysis --path_to_config config.yaml
coverage:
	coverage run -m unittest discover && coverage report -m
docker_build:
	docker image build -t latent-semantic-analysis .
docker_run:
	docker container run -it latent-semantic-analysis
clean:
	rm -rf models/model*
