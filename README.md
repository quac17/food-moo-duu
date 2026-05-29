# food-moo-duu

He thong goi y mon an thong minh chay offline bang Python, thiet ke theo kien truc 3 lop:

- Layer 1: Intent & Context Tracking (TF-IDF + ML truyen thong + Dialog State Tracking)
- Layer 2: Adaptive Recommendation Engine (Linear Weight Scoring + Hebbian Online Learning)
- Layer 3: Genetic Response Generator (Template + Genetic Algorithm + Epsilon-Greedy + Roulette)

## Project Tree

```text
food-moo-duu/
├── data/
│   ├── common_config.json
│   ├── layer1/
│   │   ├── tags.json
│   │   ├── conflict_pairs.json
│   │   ├── datasets.json
│   │   ├── dataset_v001/
│   │   │   ├── intent_samples.csv
│   │   │   └── intent_train_data.json
│   │   └── session_state.json         # runtime artifact
│   ├── layer2/
│   │   ├── datasets.json
│   │   ├── dataset_v001/
│   │   │   ├── food_weight_matrix.json
│   │   │   └── dataset_manifest.json
│   │   ├── food_weight_matrix.json    # legacy root copy
│   │   └── dishes_100.json            # legacy path
│   └── layer3/
│       ├── datasets.json
│       ├── dataset_v001/
│       │   ├── gene_pool.json
│       │   └── fitness_history.json
│       ├── gene_pool.json             # legacy root copy
│       ├── fitness_history.json       # legacy root copy
│       ├── language_genes.json
│       └── chromosome_fitness.json
├── docker/
│   ├── layer1/docker-compose.yml
│   ├── layer2/docker-compose.yml
│   └── layer3/docker-compose.yml
├── src/
│   ├── core/
│   │   ├── constants.py
│   │   └── pipeline.py
│   ├── layer1_intent_context/
│   │   ├── dialog_state.py
│   │   ├── intent_tracker.py
│   │   └── run_layer1.py
│   ├── layer2_adaptive_recommendation/
│   │   ├── online_learning.py
│   │   ├── recommendation_engine.py
│   │   └── run_layer2.py
│   └── layer3_genetic_response/
│       ├── fitness_manager.py
│       ├── genetic_generator.py
│       └── run_layer3.py
├── Dockerfile
├── Makefile
├── main.py
└── requirements.txt
```

## Quy uoc du lieu epoch

- Moi layer quan ly bo input theo version qua thu muc `dataset_v001`, `dataset_v002`, ...
- File `datasets.json` o root cua tung layer chi ra dataset dang active.
- Thanh phan co dinh (vi du: `tags.json`, `conflict_pairs.json`, `common_config.json`) dat o root, khong tach theo epoch.
- Runtime artifact (nhu `session_state.json`) khong xem la input epoch.
- Layer1 da migrate sang load theo `datasets.json` + thu muc epoch `dataset_v001`.
- Layer2/Layer3 hien van giu mot so file legacy o root de tuong thich code hien tai.

## Layer1 chat flow (export tag only)

- Flow rieng de chat va dung o Layer1:
  1) User chat
  2) IntentTracker predict tags
  3) DialogStateTracker update context
  4) Export tags ra man hinh + ghi `data/layer1/tag_exports.jsonl`
- Khong goi Layer2/Layer3 trong flow nay.

Lenh nhanh:

```bash
make layer1-chat
```

1 luot (khong interactive):

```bash
make layer1-export
```

## Kien truc va luong xu ly

```mermaid
flowchart TD
  userInput1[UserChat1] --> layer1Intent[Layer1IntentTracker]
  layer1Intent --> dstUpdate[DialogStateUpdate]
  dstUpdate --> layer2Score[Layer2LinearScoring]
  layer2Score --> layer3Text[Layer3GeneticResponse]
  layer3Text --> userInput2[UserChat2QuayXe]
  userInput2 --> layer1Intent2[Layer1IntentTracker]
  layer1Intent2 --> dstUpdate2[DialogStateUpdate]
  dstUpdate2 --> layer2Rescore[Layer2Rescoring]
  layer2Rescore --> userPick[UserSelectDish]
  userPick --> hebbianUpdate[HebbianMatrixUpdate]
  userPick --> fitnessUpdate[GeneticFitnessUpdate]
```

### Layer 1 - Intent & Context Tracking
- Nhan chat tu user.
- Dung `TfidfVectorizer + OneVsRest(LogisticRegression)` de du doan 30 tags.
- `update_context()` ap dung 3 quy luat:
  - Decay: hao mon diem theo turn.
  - Accumulation: cong don theo confidence moi.
  - Conflict Resolution: giam diem tag yeu trong cap doi nghich.
- Input train theo epoch tai `data/layer1/dataset_v001/`.
- Luu trang thai DST vao `data/layer1/session_state.json`.

### Layer 2 - Adaptive Recommendation Engine
- Tinh diem cho 100 mon an theo cong thuc:
  - `score(dish) = sum(activation[tag] * weight[dish][tag])`
- Khi user chon mon:
  - Tang trong so lien ket tag-kich-hoat (Hebbian positive).
  - Giam nhe trong so voi tag yeu de tranh drift (forgetting).
- Input canonical theo epoch tai `data/layer2/dataset_v001/food_weight_matrix.json`.
- File `data/layer2/dishes_100.json` dang duoc giu de tuong thich code hien tai.

### Layer 3 - Genetic Response Generator
- Sinh cau theo chromosome `(opening, action, closing)`.
- Chien luoc sinh:
  - Epsilon-Greedy de can bang explore/exploit.
  - Roulette Wheel Selection theo fitness.
  - Mutation them slang theo xac suat.
- Implicit feedback:
  - User chon mon -> tang fitness.
  - User khong chon/thoat -> giam fitness.
- Input genes/fitness theo epoch tai `data/layer3/dataset_v001/`.
- Root layer3 giu them alias/legacy de phuc vu chuyen doi schema.

## Chay local

```bash
python -m pip install -r requirements.txt
python main.py
```

Kich ban demo trong `main.py`:
1. User chat 1 -> goi y top mon.
2. User chat 2 (quay xe) -> cap nhat context -> goi y moi.
3. User nhap ID mon chon -> cap nhat Hebbian + Fitness.

## Lenh nhanh voi Makefile

```bash
make setup
make run
make train-layer1
make layer1
make layer2
make layer3
```

## Docker Compose theo tung layer

```bash
docker compose -f docker/layer1/docker-compose.yml up --build
docker compose -f docker/layer2/docker-compose.yml up --build
docker compose -f docker/layer3/docker-compose.yml up --build
```
