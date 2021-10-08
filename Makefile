all:
	python -m latent_semantic_analysis --path_to_config config.yaml
coverage:
	coverage run -m unittest discover && coverage report -m
docker_build:
	docker image build -t latent-semantic-analysis .
docker_run:
	docker container run -it latent-semantic-analysis
pypi_packages:
	pip install --upgrade build twine
pypi_build:
	python -m build
pypi_twine:
	python -m twine upload --repository testpypi dist/*
clean:
	rm -rf models/model*
