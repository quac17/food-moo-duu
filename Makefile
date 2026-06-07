PYTHON=python

.PHONY: setup app-run app-demo app-docker layer1-train layer1-run layer1-chat layer1-docker layer1-rl-generate layer1-rl-train layer1-rl-check layer2-run layer2-docker layer2-migrate layer2-check layer2-test layer2-reset-runtime layer3-run layer3-docker

setup:
	$(PYTHON) -m pip install -r requirements.txt

app-run:
	$(PYTHON) main.py

app-demo:
	$(PYTHON) main.py --non-interactive --top-k 5

app-docker:
	docker build -t food-moo-duu:latest . && docker run --rm food-moo-duu:latest

layer1-train:
	$(PYTHON) -m src.layer1_intent_context.intent_tracker --all-datasets

layer1-run:
	$(PYTHON) -m src.layer1_intent_context.run_layer1 --all-datasets --message "Toi dang met va muon mon nuoc am" --top-k 8 --raw-threshold 0.2 --context-threshold 0.1

layer1-chat:
	$(PYTHON) -m src.layer1_intent_context.run_layer1 --chat --all-datasets --top-k 8 --raw-threshold 0.2 --context-threshold 0.1

layer1-docker:
	docker compose -f docker/layer1/docker-compose.yml run --rm layer1-intent-context

layer1-rl-generate:
	$(PYTHON) scripts/generate_layer1_rl_feedback.py --dataset dataset_v002 --samples 80

layer1-rl-train:
	$(PYTHON) -m src.layer1_intent_context.train_reinforcement --include-simulated --threshold 0.18

layer1-rl-check:
	$(PYTHON) -c "import json, pathlib; p=pathlib.Path('data/layer1/rl_feedback/selected_dish_events.jsonl'); s=pathlib.Path('data/layer1/rl_feedback/selected_dish_events_simulated.jsonl'); print('real_events', sum(1 for _ in p.open(encoding='utf-8')) if p.exists() else 0); print('sim_events', sum(1 for _ in s.open(encoding='utf-8')) if s.exists() else 0)"

layer2-run:
	$(PYTHON) -m src.layer2_adaptive_recommendation.run_layer2

layer2-docker:
	docker compose -f docker/layer2/docker-compose.yml run --rm layer2-adaptive-recommendation

layer2-migrate:
	$(PYTHON) -m src.layer2_adaptive_recommendation.migrate_to_canonical

layer2-check:
	$(PYTHON) -m src.layer2_adaptive_recommendation.check_schema_drift

layer2-test:
	$(PYTHON) -m unittest discover -s tests -p "test_layer2*.py" -v

layer2-reset-runtime:
	$(PYTHON) -m src.layer2_adaptive_recommendation.reset_runtime

layer3-run:
	$(PYTHON) -m src.layer3_genetic_response.run_layer3 --repeat 2

layer3-docker:
	docker compose -f docker/layer3/docker-compose.yml run --rm layer3-genetic-response
