PYTHON=python

.PHONY: setup run train-layer1 layer1 layer2 layer3

setup:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) main.py

train-layer1:
	$(PYTHON) -m src.layer1_intent_context.intent_tracker

layer1:
	docker compose -f docker/layer1/docker-compose.yml up --build

layer2:
	docker compose -f docker/layer2/docker-compose.yml up --build

layer3:
	docker compose -f docker/layer3/docker-compose.yml up --build
