# Thông số đánh giá hiệu quả — Food Moo Duu

> Tài liệu tổng hợp bộ chỉ số (metric) dùng để đánh giá từng layer và pipeline tổng thể.
> Số liệu dưới đây lấy từ lần chạy mới nhất: `data/evaluation/runs/run_20260610_004458/`.

---

## 1. Cách tái tạo kết quả

```bash
# Sinh thêm dữ liệu giả lập Layer 3 (fitness + response pairs)
make layer3-simulate

# Train ablation Layer 1 + đánh giá toàn bộ + xuất folder
make eval-train-and-run

# Chỉ đánh giá (không train lại Layer 1)
make eval-run
```

Mỗi lần chạy tạo thư mục mới: `data/evaluation/runs/run_YYYYMMDD_HHMMSS/` gồm JSON, CSV, `summary.md`.

---

## 2. Thông số chạy (manifest)

| Tham số | Giá trị |
|---|---|
| Thời gian | `20260610_004458` |
| `active_dataset` | `dataset_v001` |
| `use_all_datasets` | `true` (709 mẫu intent) |
| `top_k` | 5 |
| `include_simulated` | `true` (RL events thật + giả lập) |
| `train_layer1` | `false` (chỉ đánh giá) |
| Threshold tag L1 | 0.3 |

**Dữ liệu giả lập Layer 3** (trước khi chạy eval):
- `fitness_history.json`: +50 lượt cập nhật (`source: simulated`)
- `response_train_pairs_simulated.jsonl`: 40 cặp template + feedback

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

**Diễn giải ngắn:**
- Precision cao (~0.75) nhưng recall thấp (~0.18) → mô hình **thận trọng**, ít gán tag nhưng khi gán thì thường đúng.
- Macro F1 thấp hơn micro F1 → một số tag hiếm khó học.
- Subset accuracy ~0.7% → khó khớp chính xác toàn bộ bộ tag cùng lúc (bài toán multi-label khó).

### 3.2 Metric theo tag

Chi tiết ~53 tag: `layer1/without_rl_per_tag.csv`, `layer1/with_rl_per_tag.csv`.

Ví dụ tag có F1 tốt (support > 0):
- `time_morning`: P=0.80, R=0.24, F1=0.36 (support=17)
- `time_noon`: P=0.67, R=0.25, F1=0.36 (support=8)

---

## 4. Layer 2 — Adaptive Recommendation

Ma trận đánh giá: **canonical** (`food_weight_matrix.json`), không dùng runtime Hebbian.

### 4.1 Oracle (intrinsic) — tag CSV làm context lý tưởng

**709 mẫu** | relevant = top-3 món theo điểm oracle

| Metric | @3 | @5 |
|---|---:|---:|
| Hit@K | 1.0000 | 1.0000 |
| Precision@K | 0.9093 | 0.5839 |
| Recall@K | 0.9093 | 0.9732 |
| NDCG@K | 0.9325 | 0.9708 |
| MRR | — | 1.0000 |
| Mean Rank | — | 1.0000 |

**Ý nghĩa:** Thuật toán scoring tuyến tính **khớp tốt** khi context tag đã chính xác (upper bound của L2). Không phản ánh lỗi nhận diện intent từ L1.

### 4.2 Behavioral — RL feedback events

**104 events** (4 runtime + 100 simulated) | relevant = `chosen_dish_id`

| Metric | @3 | @5 |
|---|---:|---:|
| Hit@K | 0.0288 | 0.0385 |
| Precision@K | 0.0096 | 0.0077 |
| Recall@K | 0.0288 | 0.0385 |
| NDCG@K | 0.0182 | 0.0223 |
| MRR | — | 0.0168 |
| Mean Rank | — | 5.87 |
| Avg score món chọn | — | 0.7649 |

**Ý nghĩa:** Trên feedback thực tế, top-5 chỉ trúng ~3.85% — khoảng cách lớn giữa oracle và behavioral cho thấy **context thực tế** (từ DST + chat) khó hơn nhiều so với tag CSV lý tưởng.

---

## 5. Layer 3 — Genetic Response

Nguồn: `data/layer3/dataset_v001/fitness_history.json`

### 5.1 Metric tổng thể (runtime + simulated)

| Metric | Giá trị |
|---|---:|
| Tổng số cập nhật fitness | **61** (11 runtime + 50 simulated) |
| Success rate (tổng) | **0.5246** (52.46%) |
| Failure rate | 0.4754 |
| Số chromosome unique | **33** |
| Avg fitness | **2.6106** |
| Max fitness | 25.4 |
| Min fitness | 0.05 |
| Avg gain khi success | +0.25 |

### 5.2 Tách theo nguồn dữ liệu

| Nguồn | Số lượt | Success rate |
|---|---:|---:|
| Runtime (chatbot thật) | 11 | **0.5455** (54.55%) |
| Simulated (gene pool) | 50 | **0.5200** (52.00%) |

**Dữ liệu bổ sung:** `data/layer3/dataset_v002/response_train_pairs_simulated.jsonl` — 40 cặp `(template, feedback)` dùng cho huấn luyện/kiểm tra offline.

**Top chromosome theo fitness:**
1. `mood_stressed_0_1_0` — 25.4
2. `mood_sick_0_0_1` — 18.7
3. `mood_stressed_1_0_1` — 9.8

**Ý nghĩa:** Khoảng một nửa lần user chọn món → câu thoại GA được củng cố. Sau khi bổ sung 50 mẫu giả lập, success rate ổn định hơn (~52%) và số chromosome đa dạng hơn (33). Không có ground-truth câu trả lời “đúng” — chỉ đánh giá qua implicit feedback.

**Sinh thêm data giả lập:**
```bash
make layer3-simulate
# hoặc tùy chỉnh:
python scripts/generate_layer3_simulated_fitness.py --fitness-samples 50 --pair-samples 40 --success-rate 0.58
```

---

## 6. Pipeline tổng thể (end-to-end)

Trên **104 RL events**: `context_tags` → L2 recommend → so với `chosen_dish_id`.

| Metric | Không RL | Có RL |
|---|---:|---:|
| Hit@5 | 0.0385 | 0.0385 |
| MRR | 0.0168 | 0.0168 |
| Tag overlap ratio (L1 vs event) | 0.2508 | 0.2508 |
| Avg score món chọn | 0.7649 | 0.7649 |

**Feedback runtime** (`feedback_reports.jsonl`, 8 mẫu):
- `feedback_delta_mean` = **+0.647** (điểm món tăng sau Hebbian update)

---

## 7. Bảng tổng hợp nhanh (cho slide)

| Thành phần | Chỉ số chính | Kết quả |
|---|---|---|
| L1 Intent | Macro F1 (val 142) | **0.189** |
| L1 Ablation | Delta macro F1 (RL) | **0.0** |
| L2 Oracle | Hit@5 / MRR | **1.0 / 1.0** |
| L2 Behavioral | Hit@5 / MRR | **0.039 / 0.017** |
| L3 Genetic | Success rate (61 lượt) | **52.5%** |
| L3 Runtime only | Success rate (11 lượt) | **54.6%** |
| Pipeline E2E | Hit@5 | **0.039** |
| Học online L2 | Feedback delta mean | **+0.647** |

---

## 8. Hạn chế khi báo cáo

1. **L2 oracle** dùng tag CSV hoàn hảo — chỉ đo chất lượng scoring, không đo L1.
2. **L2 behavioral** vẫn chủ yếu là simulated (100/104) — nên báo cáo tách real vs simulated.
3. **L3 simulated** (50/61) sinh từ gene pool với success rate cố định (~58%) — ổn định hơn nhưng chưa thay thế phiên chat thật.
4. **L3** không có metric NLG chuẩn (BLEU/ROUGE) — chỉ fitness ngầm.
5. **Ablation RL L1** cần chạy `make eval-train-and-run` trên môi trường train PyTorch ổn định để có delta F1 thật.
6. **Behavioral Hit@5 thấp** có thể do: context DST khác tag train, ma trận canonical chưa tối ưu cho use-case runtime, hoặc user chọn món ngoài top-K.

---

## 9. File tham chiếu

| File | Nội dung |
|---|---|
| `data/evaluation/runs/run_20260610_004458/summary.json` | Bảng số tổng hợp |
| `data/evaluation/runs/run_20260610_004458/layer1/` | Metric L1 + ablation + per-tag CSV |
| `data/evaluation/runs/run_20260610_004458/layer2/` | Oracle + behavioral |
| `data/evaluation/runs/run_20260610_004458/layer3/` | Fitness metrics (có tách runtime/simulated) |
| `data/evaluation/runs/run_20260610_004458/pipeline/` | End-to-end + feedback delta |
| `data/layer3/dataset_v002/response_train_pairs_simulated.jsonl` | Cặp response giả lập L3 |
| `scripts/generate_layer3_simulated_fitness.py` | Script sinh data giả lập L3 |
| `src/evaluation/` | Mã nguồn tính metric |
