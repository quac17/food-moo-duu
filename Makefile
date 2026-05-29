PYTHON=python

.PHONY: setup run run-demo train-layer1 layer1-chat layer1-export layer1 layer2 layer3 docker-main

setup:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) main.py

run-demo:
	$(PYTHON) main.py --non-interactive --top-k 5

train-layer1:
	$(PYTHON) -m src.layer1_intent_context.intent_tracker

layer1-chat:
	$(PYTHON) -m src.layer1_intent_context.run_layer1 --chat --top-k 8 --threshold 0.2

layer1-export:
	$(PYTHON) -m src.layer1_intent_context.run_layer1 --message "Toi dang met va muon mon nuoc am" --top-k 8 --threshold 0.2

docker-main:
	docker build -t food-moo-duu:latest . && docker run --rm food-moo-duu:latest

layer1:
	docker compose -f docker/layer1/docker-compose.yml up --build

layer2:
	docker compose -f docker/layer2/docker-compose.yml up --build

layer3:
	docker compose -f docker/layer3/docker-compose.yml up --build
