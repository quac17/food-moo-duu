# Thông số đánh giá hiệu quả — Food Moo Duu

> Tài liệu tổng hợp bộ chỉ số (metric) dùng để đánh giá từng layer và pipeline tổng thể.
> Số liệu dưới đây lấy từ lần chạy mới nhất: `data/evaluation/runs/run_20260610_014807/`.
> Giải thích ý nghĩa từng metric: [`docs/giai_thich_metrics.txt`](giai_thich_metrics.txt).

---

## 1. Cách tái tạo kết quả

```bash
# Gan lai tag menu Layer 2 + migrate canonical
python scripts/rebuild_dish_catalog.py
make layer2-migrate

# Reset runtime/state sau cap nhat catalog
make layer1-reset-state
make layer2-reset-runtime

# Sinh them du lieu gia lap Layer 3
make layer3-simulate

# Chi danh gia (khong train lai Layer 1)
make eval-run

# Train ablation Layer 1 + danh gia toan bo
make eval-train-and-run

# Cap nhat hinh + slide
python scripts/build_diagrams.py
python scripts/build_slides.py
```

Mỗi lần chạy tạo thư mục mới: `data/evaluation/runs/run_YYYYMMDD_HHMMSS/` gồm JSON, CSV, `summary.md`.

---

## 2. Thông số chạy (manifest)

| Tham số | Giá trị |
|---|---|
| Thời gian | `20260610_014807` |
| `active_dataset` | `dataset_v001` |
| `use_all_datasets` | `true` (709 mẫu intent) |
| `top_k` | 5 |
| `include_simulated` | `true` |
| `train_layer1` | `true` (fallback do train PyTorch loi tren Windows) |
| Threshold tag L1 | 0.3 |
| Menu Layer 2 | **120 món** (9 đồ uống), tag đã gán lại qua `scripts/rebuild_dish_catalog.py` |

### DST hyperparameters (runtime)

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `context_decay` | **0.55** | Session state phai nhanh — ưu tiên lượt chat hiện tại |
| `context_accumulation_alpha` | **0.88** | Raw tag mạnh hơn khi cộng vào context |
| `context_conflict_beta` | **0.40** | Giảm tag đối nghịch khi user đổi ý |

Cấu hình: `data/common_config.json`. **Raw intent** = chỉ câu hiện tại; **Context tags** = sau DST, dùng cho L2 recommend và export mặc định.

**Lưu ý đánh giá:** Sau reset runtime Layer 2, `feedback_delta_mean = 0`. Metric L2 behavioral/E2E vẫn đọc `context_tags` từ RL events cũ — cần thu thập phiên chat mới để phản ánh catalog 120 món.

---

## 3. Layer 1 — Intent Recognition (multi-label)

**Tập đánh giá:** validation 20% từ `intent_samples.csv` (all datasets), `random_seed=42` → **142 mẫu**.

### 3.1 Metric tổng thể

| Metric | Không RL | Có RL | Delta |
|---|---:|---:|---:|
| Micro F1 | 0.2909 | 0.2909 | 0.0 |
| Macro F1 | 0.1890 | 0.1890 | 0.0 |
| Micro Precision | 0.7477 | 0.7477 | 0.0 |
| Micro Recall | 0.1806 | 0.1806 | 0.0 |
| Hamming Loss | 0.0518 | 0.0518 | — |
| Subset Accuracy | 0.0070 | 0.0070 | 0.0 |

**Diễn giải:** Precision cao (~0.75), recall thấp (~0.18) — mô hình thận trọng khi gán tag. Train Layer 1 trong `eval-train-and-run` bị fallback do crash PyTorch trên môi trường hiện tại; metric dùng artifact cũ.

---

## 4. Layer 2 — Adaptive Recommendation

Ma trận đánh giá: **canonical** (`food_weight_matrix.json`, **120 món**).

### 4.1 Oracle (intrinsic) — tag CSV lý tưởng

**709 mẫu** | relevant = top-3 món theo điểm oracle

| Metric | @3 | @5 |
|---|---:|---:|
| Hit@K | 1.0000 | 1.0000 |
| Precision@K | 0.7673 | 0.5444 |
| Recall@K | 0.7673 | 0.9074 |
| NDCG@K | 0.7974 | 0.8777 |
| MRR | — | 0.9492 |
| Mean Rank | — | 1.1016 |

### 4.2 Behavioral — RL feedback events

**105 events** (5 runtime + 100 simulated) | relevant = `chosen_dish_id`

| Metric | @3 | @5 |
|---|---:|---:|
| Hit@K | 0.0095 | **0.0190** |
| Precision@K | 0.0032 | 0.0038 |
| Recall@K | 0.0095 | 0.0190 |
| NDCG@K | 0.0060 | 0.0101 |
| MRR | — | **0.0071** |
| Mean Rank | — | 5.94 |
| Avg score món chọn | — | 1.2893 |

**Ý nghĩa:** Catalog mới cải thiện oracle MRR (0.95) nhưng behavioral vẫn thấp vì RL events map với ma trận/runtime cũ. Cần log thêm phiên sau khi reset.

---

## 5. Layer 3 — Genetic Response

Nguồn: `data/layer3/dataset_v001/fitness_history.json`

### 5.1 Metric tổng thể

| Metric | Giá trị |
|---|---:|
| Tổng số cập nhật fitness | **113** (13 runtime + 100 simulated) |
| Success rate (tổng) | **0.5221** (52.21%) |
| Số chromosome unique | **35** |
| Avg fitness | **2.5971** |
| Max fitness | 25.4 |
| Min fitness | 0.05 |

### 5.2 Tách theo nguồn

| Nguồn | Số lượt | Success rate |
|---|---:|---:|
| Runtime | 13 | **53.85%** |
| Simulated | 100 | **52.00%** |

---

## 6. Pipeline tổng thể (end-to-end)

Trên **105 RL events**:

| Metric | Không RL | Có RL |
|---|---:|---:|
| Hit@5 | **0.0190** | **0.0190** |
| MRR | 0.0071 | 0.0071 |
| Tag overlap ratio | 0.2484 | 0.2484 |
| Avg score món chọn | 1.2893 | 1.2893 |

**Feedback runtime** (`feedback_reports.jsonl`):
- `feedback_delta_mean` = **0.0** (đã reset runtime sau cập nhật catalog)

---

## 7. Bảng tổng hợp nhanh (cho slide)

| Thành phần | Chỉ số chính | Kết quả |
|---|---|---|
| DST runtime | decay / alpha / beta | **0.55 / 0.88 / 0.4** |
| L1 Intent | Macro F1 (val 142) | **0.189** |
| L2 Oracle | Hit@5 / MRR | **1.0 / 0.949** |
| L2 Behavioral | Hit@5 / MRR | **0.019 / 0.007** |
| L3 Genetic | Success rate (113 lượt) | **52.2%** |
| L3 Runtime only | Success rate (13 lượt) | **53.9%** |
| Pipeline E2E | Hit@5 | **0.019** |
| Học online L2 | Feedback delta mean | **0.0** (sau reset) |

---

## 8. Hạn chế khi báo cáo

1. **L2 oracle** dùng tag CSV hoàn hảo — không đo L1 + DST runtime.
2. **L2 behavioral** chủ yếu simulated (100/105); `context_tags` log từ phiên cũ.
3. **Train Layer 1** (`make layer1-train`) crash trên Windows hiện tại — cần môi trường PyTorch ổn định để retrain.
4. **L3** không có metric NLG chuẩn — chỉ implicit feedback.
5. **Catalog 120 món** đã gán lại tag; demo chat mới sẽ gợi ý lẩu/nhậu đúng hơn so với ma trận cũ.

---

## 9. File tham chiếu

| File | Nội dung |
|---|---|
| `data/common_config.json` | DST hyperparameters |
| `data/layer2/dishes_100.json` | Catalog 120 món + tag |
| `scripts/rebuild_dish_catalog.py` | Script gán lại tag + bổ sung món |
| `data/evaluation/runs/run_20260610_014807/summary.json` | Bảng số tổng hợp |
| `data/evaluation/runs/run_20260610_014807/manifest.json` | DST + tham số chạy |
| `data/evaluation/runs/run_20260610_014807/layer1/` | Metric L1 |
| `data/evaluation/runs/run_20260610_014807/layer2/` | Oracle + behavioral |
| `data/evaluation/runs/run_20260610_014807/layer3/` | Fitness metrics |
| `data/evaluation/runs/run_20260610_014807/pipeline/` | End-to-end |
| `src/evaluation/` | Mã nguồn tính metric |
| `docs/giai_thich_metrics.txt` | Giải thích ý nghĩa các metric (file txt) |
