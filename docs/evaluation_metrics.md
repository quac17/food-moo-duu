# Thông số đánh giá hiệu quả — Food Moo Duu

> Tài liệu tổng hợp bộ chỉ số (metric) dùng để đánh giá từng layer và pipeline tổng thể.
> Số liệu dưới đây lấy từ lần chạy mới nhất: `data/evaluation/runs/run_20260610_011454/`.
> Giải thích ý nghĩa từng metric: [`docs/giai_thich_metrics.txt`](giai_thich_metrics.txt).

---

## 1. Cách tái tạo kết quả

```bash
# Sinh thêm dữ liệu giả lập Layer 3 (fitness + response pairs)
make layer3-simulate

# Chỉ đánh giá (không train lại Layer 1)
make eval-run

# Train ablation Layer 1 + đánh giá toàn bộ
make eval-train-and-run
```

Mỗi lần chạy tạo thư mục mới: `data/evaluation/runs/run_YYYYMMDD_HHMMSS/` gồm JSON, CSV, `summary.md`.

---

## 2. Thông số chạy (manifest)

| Tham số | Giá trị |
|---|---|
| Thời gian | `20260610_011454` |
| `active_dataset` | `dataset_v001` |
| `use_all_datasets` | `true` (709 mẫu intent) |
| `top_k` | 5 |
| `include_simulated` | `true` |
| `train_layer1` | `false` |
| Threshold tag L1 | 0.3 |

### DST hyperparameters (runtime)

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `context_decay` | **0.55** | Session state phai nhanh — ưu tiên lượt chat hiện tại |
| `context_accumulation_alpha` | **0.88** | Raw tag mạnh hơn khi cộng vào context |
| `context_conflict_beta` | **0.40** | Giảm tag đối nghịch khi user đổi ý |

Cấu hình: `data/common_config.json`. **Raw intent** = chỉ câu hiện tại; **Context tags** = sau DST, dùng cho L2 recommend và export mặc định.

**Lưu ý đánh giá:** Metric L2 behavioral/E2E đọc `context_tags` đã log trong RL events — phần lớn ghi trước khi tinh chỉnh DST. Phiên chat mới sau chỉnh DST sẽ có context khác; cần thu thập thêm RL events để phản ánh đầy đủ.

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

**Diễn giải:** Precision cao (~0.75), recall thấp (~0.18) — mô hình thận trọng khi gán tag. Metric L1 **không phụ thuộc DST** (chỉ đo raw predict trên validation CSV).

---

## 4. Layer 2 — Adaptive Recommendation

Ma trận đánh giá: **canonical** (`food_weight_matrix.json`).

### 4.1 Oracle (intrinsic) — tag CSV lý tưởng

**709 mẫu** | relevant = top-3 món theo điểm oracle

| Metric | @3 | @5 |
|---|---:|---:|
| Hit@K | 1.0000 | 1.0000 |
| Precision@K | 0.9093 | 0.5839 |
| Recall@K | 0.9093 | 0.9732 |
| NDCG@K | 0.9325 | 0.9708 |
| MRR | — | 1.0000 |
| Mean Rank | — | 1.0000 |

### 4.2 Behavioral — RL feedback events

**105 events** (5 runtime + 100 simulated) | relevant = `chosen_dish_id`

| Metric | @3 | @5 |
|---|---:|---:|
| Hit@K | 0.0381 | **0.0476** |
| Precision@K | 0.0127 | 0.0095 |
| Recall@K | 0.0381 | 0.0476 |
| NDCG@K | 0.0240 | 0.0281 |
| MRR | — | **0.0214** |
| Mean Rank | — | 5.83 |
| Avg score món chọn | — | 0.7570 |

**Ý nghĩa:** Khoảng cách oracle vs behavioral vẫn lớn — context runtime (DST) khó hơn tag CSV lý tưởng. Sau tinh chỉnh DST, cần log thêm phiên chat để cập nhật metric behavioral.

---

## 5. Layer 3 — Genetic Response

Nguồn: `data/layer3/dataset_v001/fitness_history.json`

### 5.1 Metric tổng thể

| Metric | Giá trị |
|---|---:|
| Tổng số cập nhật fitness | **62** (12 runtime + 50 simulated) |
| Success rate (tổng) | **0.5323** (53.23%) |
| Số chromosome unique | **34** |
| Avg fitness | **2.5706** |
| Max fitness | 25.4 |
| Min fitness | 0.05 |

### 5.2 Tách theo nguồn

| Nguồn | Số lượt | Success rate |
|---|---:|---:|
| Runtime | 12 | **58.33%** |
| Simulated | 50 | **52.00%** |

---

## 6. Pipeline tổng thể (end-to-end)

Trên **105 RL events**:

| Metric | Không RL | Có RL |
|---|---:|---:|
| Hit@5 | **0.0476** | **0.0476** |
| MRR | 0.0214 | 0.0214 |
| Tag overlap ratio | 0.2508 | 0.2508 |
| Avg score món chọn | 0.7570 | 0.7570 |

**Feedback runtime** (`feedback_reports.jsonl`):
- `feedback_delta_mean` = **+0.599** (Hebbian update)

---

## 7. Bảng tổng hợp nhanh (cho slide)

| Thành phần | Chỉ số chính | Kết quả |
|---|---|---|
| DST runtime | decay / alpha / beta | **0.55 / 0.88 / 0.4** |
| L1 Intent | Macro F1 (val 142) | **0.189** |
| L2 Oracle | Hit@5 / MRR | **1.0 / 1.0** |
| L2 Behavioral | Hit@5 / MRR | **0.048 / 0.021** |
| L3 Genetic | Success rate (62 lượt) | **53.2%** |
| L3 Runtime only | Success rate (12 lượt) | **58.3%** |
| Pipeline E2E | Hit@5 | **0.048** |
| Học online L2 | Feedback delta mean | **+0.599** |

---

## 8. Hạn chế khi báo cáo

1. **L2 oracle** dùng tag CSV hoàn hảo — không đo L1 + DST runtime.
2. **L2 behavioral** chủ yếu simulated (100/105); `context_tags` log có thể từ DST cũ.
3. **DST mới** (decay 0.55) chỉ áp dụng khi chạy chat/pipeline — chưa phản ánh hết trong RL events cũ.
4. **L3** không có metric NLG chuẩn — chỉ implicit feedback.
5. **Ablation RL L1** cần `make eval-train-and-run` trên môi trường train PyTorch ổn định.

---

## 9. File tham chiếu

| File | Nội dung |
|---|---|
| `data/common_config.json` | DST hyperparameters |
| `data/evaluation/runs/run_20260610_011454/summary.json` | Bảng số tổng hợp |
| `data/evaluation/runs/run_20260610_011454/manifest.json` | DST + tham số chạy |
| `data/evaluation/runs/run_20260610_011454/layer1/` | Metric L1 |
| `data/evaluation/runs/run_20260610_011454/layer2/` | Oracle + behavioral |
| `data/evaluation/runs/run_20260610_011454/layer3/` | Fitness metrics |
| `data/evaluation/runs/run_20260610_011454/pipeline/` | End-to-end |
| `src/evaluation/` | Mã nguồn tính metric |
| `docs/giai_thich_metrics.txt` | Giải thích ý nghĩa các metric (file txt) |
