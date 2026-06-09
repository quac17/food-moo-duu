PYTHON=python
LAYER1_RUN_MESSAGE ?= $(shell $(PYTHON) -c "import random; samples=['Sang nay toi muon bua nhe de de tieu','Trua nay toi can bua com can bang va no lau','Chieu nay toi hoi met muon mon nuoc am','Toi nay toi di mot minh muon mon am ap','Troi nong nen toi muon mon mat va it beo','Dem muon toi chi muon an nhe truoc khi ngu']; print(random.choice(samples))")

.PHONY: setup app-run app-demo app-docker app-run-docker app-demo-docker eval-run eval-train-and-run layer1-train layer1-train-active layer1-run layer1-chat layer1-reset-state layer1-docker layer1-train-docker layer1-run-docker layer1-chat-docker layer1-reset-state-docker layer1-rl-generate layer1-rl-train layer1-rl-check layer1-rl-generate-docker layer1-rl-train-docker layer1-rl-check-docker layer2-run layer2-docker layer2-migrate layer2-check layer2-test layer2-reset-runtime layer2-run-docker layer2-migrate-docker layer2-check-docker layer2-test-docker layer2-reset-runtime-docker layer3-simulate layer3-run layer3-docker layer3-run-docker

setup:
	$(PYTHON) -m pip install -r requirements.txt

app-run:
	$(PYTHON) main.py

app-demo:
	$(PYTHON) main.py --non-interactive --top-k 5

app-docker:
	docker compose -f docker/app/docker-compose.yml run --rm app

app-run-docker:
	docker compose -f docker/app/docker-compose.yml run --rm app python main.py

app-demo-docker:
	docker compose -f docker/app/docker-compose.yml run --rm app python main.py --non-interactive --top-k 5

eval-run:
	$(PYTHON) -m src.evaluation.run_evaluation --all-datasets --top-k 5 --include-simulated --skip-train

eval-train-and-run:
	$(PYTHON) -m src.evaluation.run_evaluation --all-datasets --top-k 5 --include-simulated

layer1-train:
	$(PYTHON) -m src.layer1_intent_context.intent_tracker --all-datasets --threshold 0.3

layer1-train-active:
	$(PYTHON) -m src.layer1_intent_context.intent_tracker --threshold 0.3

layer1-run:
	$(PYTHON) -m src.layer1_intent_context.run_layer1 --all-datasets --message "$(LAYER1_RUN_MESSAGE)" --top-k 8 --raw-threshold 0.2 --context-threshold 0.1

layer1-chat:
	$(PYTHON) -m src.layer1_intent_context.run_layer1 --chat --all-datasets --top-k 8 --raw-threshold 0.2 --context-threshold 0.1

layer1-reset-state:
	$(PYTHON) -c "import json, pathlib; tags=json.loads(pathlib.Path('data/layer1/tags.json').read_text(encoding='utf-8')).get('tag_ids', []); payload={'tag_scores': {tag: 0.0 for tag in tags}, 'turn_index': 0}; pathlib.Path('data/layer1/session_state.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'); print('Reset layer1 session_state.json ve trang thai sau train (all tags = 0).')"

layer1-docker:
	docker compose -f docker/layer1/docker-compose.yml run --rm layer1-intent-context

layer1-train-docker:
	docker compose -f docker/layer1/docker-compose.yml run --rm layer1-intent-context python -m src.layer1_intent_context.intent_tracker --all-datasets --threshold 0.3

layer1-run-docker:
	docker compose -f docker/layer1/docker-compose.yml run --rm layer1-intent-context python -m src.layer1_intent_context.run_layer1 --all-datasets --message "$(LAYER1_RUN_MESSAGE)" --top-k 8 --raw-threshold 0.2 --context-threshold 0.1

layer1-chat-docker:
	docker compose -f docker/layer1/docker-compose.yml run --rm layer1-intent-context python -m src.layer1_intent_context.run_layer1 --chat --all-datasets --top-k 8 --raw-threshold 0.2 --context-threshold 0.1

layer1-reset-state-docker:
	docker compose -f docker/layer1/docker-compose.yml run --rm layer1-intent-context python -c "import json, pathlib; tags=json.loads(pathlib.Path('data/layer1/tags.json').read_text(encoding='utf-8')).get('tag_ids', []); payload={'tag_scores': {tag: 0.0 for tag in tags}, 'turn_index': 0}; pathlib.Path('data/layer1/session_state.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'); print('Reset layer1 session_state.json ve trang thai sau train (all tags = 0).')"

layer1-rl-generate:
	$(PYTHON) scripts/generate_layer1_rl_feedback.py --dataset dataset_v002 --samples 80

layer1-rl-train:
	$(PYTHON) -m src.layer1_intent_context.train_reinforcement --include-simulated --threshold 0.18

layer1-rl-check:
	$(PYTHON) -c "import json, pathlib; p=pathlib.Path('data/layer1/rl_feedback/selected_dish_events.jsonl'); s=pathlib.Path('data/layer1/rl_feedback/selected_dish_events_simulated.jsonl'); print('real_events', sum(1 for _ in p.open(encoding='utf-8')) if p.exists() else 0); print('sim_events', sum(1 for _ in s.open(encoding='utf-8')) if s.exists() else 0)"

layer1-rl-generate-docker:
	docker compose -f docker/layer1/docker-compose.yml run --rm layer1-intent-context python scripts/generate_layer1_rl_feedback.py --dataset dataset_v002 --samples 80

layer1-rl-train-docker:
	docker compose -f docker/layer1/docker-compose.yml run --rm layer1-intent-context python -m src.layer1_intent_context.train_reinforcement --include-simulated --threshold 0.18

layer1-rl-check-docker:
	docker compose -f docker/layer1/docker-compose.yml run --rm layer1-intent-context python -c "import pathlib; p=pathlib.Path('data/layer1/rl_feedback/selected_dish_events.jsonl'); s=pathlib.Path('data/layer1/rl_feedback/selected_dish_events_simulated.jsonl'); print('real_events', sum(1 for _ in p.open(encoding='utf-8')) if p.exists() else 0); print('sim_events', sum(1 for _ in s.open(encoding='utf-8')) if s.exists() else 0)"

layer2-run:
	$(PYTHON) -m src.layer2_adaptive_recommendation.run_layer2

layer2-docker:
	docker compose -f docker/layer2/docker-compose.yml run --rm layer2-adaptive-recommendation

layer2-run-docker:
	docker compose -f docker/layer2/docker-compose.yml run --rm layer2-adaptive-recommendation python -m src.layer2_adaptive_recommendation.run_layer2

layer2-migrate:
	$(PYTHON) -m src.layer2_adaptive_recommendation.migrate_to_canonical

layer2-migrate-docker:
	docker compose -f docker/layer2/docker-compose.yml run --rm layer2-adaptive-recommendation python -m src.layer2_adaptive_recommendation.migrate_to_canonical

layer2-check:
	$(PYTHON) -m src.layer2_adaptive_recommendation.check_schema_drift

layer2-check-docker:
	docker compose -f docker/layer2/docker-compose.yml run --rm layer2-adaptive-recommendation python -m src.layer2_adaptive_recommendation.check_schema_drift

layer2-test:
	$(PYTHON) -m unittest discover -s tests -p "test_layer2*.py" -v

layer2-test-docker:
	docker compose -f docker/layer2/docker-compose.yml run --rm layer2-adaptive-recommendation python -m unittest discover -s tests -p "test_layer2*.py" -v

layer2-reset-runtime:
	$(PYTHON) -m src.layer2_adaptive_recommendation.reset_runtime

layer2-reset-runtime-docker:
	docker compose -f docker/layer2/docker-compose.yml run --rm layer2-adaptive-recommendation python -m src.layer2_adaptive_recommendation.reset_runtime

layer3-simulate:
	$(PYTHON) scripts/generate_layer3_simulated_fitness.py --fitness-samples 50 --pair-samples 40

layer3-run:
	$(PYTHON) -m src.layer3_genetic_response.run_layer3 --repeat 2

layer3-docker:
	docker compose -f docker/layer3/docker-compose.yml run --rm layer3-genetic-response

layer3-run-docker:
	docker compose -f docker/layer3/docker-compose.yml run --rm layer3-genetic-response python -m src.layer3_genetic_response.run_layer3 --repeat 2
