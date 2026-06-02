# Layer 2 Change Log

## Muc tieu dot thay doi

Hoan thien Layer 2 theo huong:

- dong bo taxonomy tag voi Layer 1
- uu tien dataset canonical
- tach runtime data khoi dataset goc
- bo sung test va drift check
- them bao cao va log cho feedback learning

## Cac thay doi da thuc hien

### 1. Chuan hoa loader cua Layer 2

File: `src/layer2_adaptive_recommendation/recommendation_engine.py`

- Doc `active_dataset` tu `data/layer2/datasets.json`
- Doc duong dan canonical va legacy tu `data/layer2/dataset_v001/dataset_manifest.json`
- Ho tro 3 nguon du lieu:
  - runtime: `data/layer2/runtime/dataset_v001_dishes_runtime.json`
  - canonical: `data/layer2/dataset_v001/food_weight_matrix.json`
  - legacy: `data/layer2/dishes_100.json`
- Thu tu uu tien hien tai:
  - runtime neu da co hoc online
  - canonical neu runtime chua ton tai
  - legacy neu canonical khong san sang

### 2. Dong bo taxonomy tag giua Layer 1 va Layer 2

File: `src/layer2_adaptive_recommendation/recommendation_engine.py`

- Them bang `LEGACY_TAG_ALIASES` de map cac tag cu ve bo tag chuan cua Layer 1
- Chuan hoa `tag_weights` ve dung bo `ALL_TAGS`
- Loai bo cac tag la khong thuoc taxonomy chuan
- Clamp weight trong khoang `[-1.0, 1.0]`

Ket qua:

- `unknown_tag_count` cua `data/layer2/dishes_100.json` da ve `0`

### 3. Migrate Layer 2 sang canonical dataset

File moi: `src/layer2_adaptive_recommendation/migrate_to_canonical.py`

- Them script migrate tu legacy `dishes_100.json` sang canonical `food_weight_matrix.json`
- Da migrate du 100 mon sang file canonical

File du lieu da thay doi:

- `data/layer2/dataset_v001/food_weight_matrix.json`
- `data/layer2/dishes_100.json`

### 4. Tach runtime khoi dataset goc

File: `src/layer2_adaptive_recommendation/recommendation_engine.py`

- Khi feedback hoc online xay ra, Layer 2 khong ghi de vao canonical nua
- Runtime state duoc ghi rieng vao:
  - `data/layer2/runtime/dataset_v001_dishes_runtime.json`

Muc dich:

- giu nguyen dataset goc de demo/train
- tranh drift du lieu goc sau nhieu lan chay

### 5. Bo sung drift checker cho Layer 2

File moi: `src/layer2_adaptive_recommendation/check_schema_drift.py`

- Kiem tra source dang duoc dung
- Kiem tra so luong mon
- Kiem tra id rong
- Kiem tra dish co thieu truong bat buoc hay khong
- Kiem tra co tag nao nam ngoai `ALL_TAGS` hay khong

Trang thai hien tai:

- Drift check pass
- `invalid_tag_key_count = 0`

### 6. Bo sung reset runtime

File moi: `src/layer2_adaptive_recommendation/reset_runtime.py`

- Xoa runtime file cua Layer 2
- Xoa feedback log file neu ton tai

### 7. Cap nhat demo context cua Layer 2

File: `src/layer2_adaptive_recommendation/run_layer2.py`

- Doi context demo tu tag legacy `time_quick_meal`
- Sang tag chuan `pref_convenient`

### 8. Bo sung unit test cho Layer 2

File moi: `tests/test_layer2.py`

Bao phu cac hanh vi:

- normalize tag legacy sang tag chuan
- uu tien canonical khi san sang
- tie-break recommend on dinh
- hebbian update duoc persist vao runtime

### 9. Bo sung integration test cho hoc online

File moi: `tests/test_layer2_integration.py`

Bao phu cac hanh vi:

- recommend truoc feedback
- feedback cap nhat dung mon da chon
- runtime file duoc tao
- canonical file giu nguyen
- engine moi uu tien runtime sau khi hoc

### 10. Them bao cao tac dong feedback trong CLI

Files:

- `src/core/pipeline.py`
- `main.py`

Thay doi:

- Them `FeedbackReport` gom:
  - `chosen_dish_id`
  - `chosen_dish_name`
  - `score_before`
  - `score_after`
  - `delta`
- Sau khi user chon mon, CLI in them bao cao:
  - diem truoc update
  - diem sau update
  - do lech delta

### 11. Them feedback log dang JSONL

File log moi:

- `data/layer2/runtime/feedback_reports.jsonl`

File code lien quan:

- `main.py`
- `src/layer2_adaptive_recommendation/reset_runtime.py`
- `tests/test_feedback_logging.py`

Noi dung moi dong log:

- `timestamp`
- `dish_id`
- `dish_name`
- `score_before`
- `score_after`
- `delta`
- `top_context`

### 12. Cap nhat Makefile

File: `Makefile`

Them cac target moi:

- `layer2-migrate`
- `layer2-check`
- `layer2-test`
- `layer2-reset-runtime`

## Validation da chay

Da chay thanh cong cac buoc sau:

- `python -m unittest discover -s tests -p "test_layer2*.py" -v`
- `python -m unittest discover -s tests -p "test_feedback_logging.py" -v`
- `python -m src.layer2_adaptive_recommendation.check_schema_drift`
- `python -m src.layer2_adaptive_recommendation.migrate_to_canonical`
- `python main.py --non-interactive --top-k 3`

Ket qua:

- test pass
- drift check pass
- main run pass
- Layer 2 runtime hoat dong dung

## File moi duoc tao

- `LAYER2_CHANGES.md`
- `src/layer2_adaptive_recommendation/migrate_to_canonical.py`
- `src/layer2_adaptive_recommendation/check_schema_drift.py`
- `src/layer2_adaptive_recommendation/reset_runtime.py`
- `tests/test_layer2.py`
- `tests/test_layer2_integration.py`
- `tests/test_feedback_logging.py`
- `data/layer2/runtime/dataset_v001_dishes_runtime.json`
- `data/layer2/runtime/feedback_reports.jsonl`

## File da sua

- `Makefile`
- `main.py`
- `src/core/pipeline.py`
- `src/layer2_adaptive_recommendation/recommendation_engine.py`
- `src/layer2_adaptive_recommendation/run_layer2.py`
- `data/layer2/dataset_v001/food_weight_matrix.json`
- `data/layer2/dishes_100.json`
- `data/layer1/session_state.json` (doi do chay pipeline trong qua trinh test)

## Trang thai hien tai cua Layer 2

Layer 2 hien da dat muc:

- chay duoc
- co test
- co drift check
- co runtime isolation
- co logging cho feedback
- co migrate canonical

Chua hoan tat tuyet doi neu muon them:

- negative feedback day du cho cac mon duoc goi y nhung khong duoc chon
- them lenh xem log nhanh trong Makefile
- them test cho nhieu chu ky feedback dai hon