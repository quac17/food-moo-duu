# Khung Slide Báo Cáo — Food Moo Duu

> Hệ thống gợi ý món ăn thông minh offline theo kiến trúc 3 layer (NLP tiếng Việt).
> Khung ~24 slide, soạn dựa trên cấu trúc code, README, pipeline đánh giá và luồng runtime thực tế của repo.

**Phân công nhóm:**
- **Trần Việt Hoàng** — Layer 1 (Tag Define) + Kiến trúc tổng thể
- **Bùi Đức Anh** — Layer 2 (Adaptive Recommendation)
- **Trần Bình Minh** — Layer 3 (Genetic Response)

**Hình minh họa** (sinh bằng `scripts/build_diagrams.py`, lưu tại `docs/images/`):
- `system_architecture.png` — kiến trúc tổng thể 3 layer + feedback loop
- `layer1_logic.png` — logic Layer 1 (Encoder → heads + DST)
- `layer2_logic.png` — logic Layer 2 (Linear scoring + Hebbian)
- `layer3_logic.png` — logic Layer 3 (Genetic Algorithm)
- `evaluation_metrics.png` — bảng tổng hợp metric đánh giá hiệu quả

---

## Slide 1 — Trang bìa

**Tiêu đề:** Food Moo Duu — Hệ thống gợi ý món ăn thông minh offline

- Slogan: *"Là bò, sáng phải rống — Là người, sống phải ráng"*
- Môn học / nhóm / 3 thành viên & phân công / ngày báo cáo
- Ngôn ngữ: Python, NLP tiếng Việt

---

## Slide 2 — Vấn đề & mục tiêu *(Hoàng)*

**Vấn đề:**
- Người dùng mô tả nhu cầu ăn uống bằng ngôn ngữ tự nhiên, đa chiều (thời gian, thời tiết, tâm trạng, khẩu vị…)
- Cần hệ thống **offline**, không phụ thuộc API cloud
- Cần **học từ phản hồi** khi người dùng chọn hoặc bỏ qua gợi ý

**Mục tiêu:**
- Xây dựng chatbot gợi ý món ăn theo kiến trúc 3 tầng
- Theo dõi ngữ cảnh hội thoại nhiều lượt (multi-turn)
- Tự cập nhật mô hình gợi ý và câu trả lời theo feedback ngầm

---

## Slide 3 — Tổng quan giải pháp *(Hoàng)*

**Ý tưởng cốt lõi:** Pipeline xử lý tuần tự 3 layer, kết hợp DL + học trực tuyến + thuật toán di truyền.

**Điểm nổi bật:**
- Offline end-to-end qua `main.py` → `FoodSuggestionPipeline`
- Mỗi layer chạy độc lập được (train/test/docker riêng)
- Dữ liệu train và runtime tách biệt rõ ràng

---

## Slide 4 — Kiến trúc hệ thống (hình minh họa bắt buộc) *(Hoàng)*

> **Hình:** `docs/images/system_architecture.png`

**Diễn giải sơ đồ:**
- User Input → Layer 1 (Intent + DST) → Layer 2 (Scoring) → Layer 3 (Response) → User
- Hai vòng phản hồi: **Hebbian update** (L2) và **Fitness update** (L3) sau khi user chọn/bỏ qua món
- Vòng **RL offline** cho Layer 1 (log feedback → train tăng cường định kỳ)

**Công nghệ:** Python, PyTorch, scikit-learn, pandas, Docker, Makefile

**Cấu trúc thư mục chính:**

| Thành phần | Vai trò |
|---|---|
| `src/core/pipeline.py` | Điều phối toàn bộ luồng |
| `src/layer1_intent_context/` | Nhận diện intent + dialog state |
| `src/layer2_adaptive_recommendation/` | Gợi ý + Hebbian learning |
| `src/layer3_genetic_response/` | Sinh câu trả lời + fitness |

---

# === PHÂN MỤC: LAYER 1 — Trần Việt Hoàng ===

## Slide 5 — [Section] Layer 1: Intent & Dialog State Tracking *(Hoàng)*

Slide phân mục mở đầu phần Layer 1.

---

## Slide 6 — Layer 1: Ý tưởng & logic thực hiện (hình minh họa) *(Hoàng)*

> **Hình:** `docs/images/layer1_logic.png`

**Ý tưởng:**
- Biến câu nói tự nhiên thành **bộ tag ngữ cảnh** (multi-label) để các layer sau dùng được
- Không chỉ phân loại đúng tag, mà còn **học không gian embedding**: câu cùng ngữ cảnh thì gần nhau

**Logic xử lý (1 lượt chat):**
1. Chuẩn hóa tiếng Việt (bỏ dấu, lowercase) + tokenize → ghép `raw || normalized`
2. **Encoder vi-SBERT** mã hóa câu (mean pooling theo attention mask)
3. Hai head song song:
   - **Classification head** → `sigmoid(logits)` = score [0,1] cho ~53 tag
   - **Projection head** → vector chuẩn hóa để tính cosine similarity
4. Áp threshold 0.3 → ra `raw_scores`

**Hàm loss kết hợp:** `BCEWithLogitsLoss` (multi-label) + `α · metric_loss` (α=0.35), với similarity target = overlap_tag / max(tag_a, tag_b).

---

## Slide 7 — Layer 1: Dialog State Tracking (logic DST) *(Hoàng)*

**Vai trò:** tích lũy ngữ cảnh qua nhiều lượt, xử lý người dùng "đổi ý".

**3 quy luật cập nhật state** (`DialogStateTracker.update_context`, đọc từ `common_config.json`):
1. **Decay:** `score *= 0.55` mỗi lượt — session state phai nhanh, ưu tiên ngữ cảnh lượt hiện tại
2. **Accumulation:** `score += 0.88 · confidence` — raw tag mạnh hơn, cộng gần đầy đủ điểm intent mới
3. **Conflict resolution:** với cặp tag đối nghịch, giảm bên yếu theo gap (`β=0.4`)
   - VD conflict pairs: `weather_hot ↔ weather_cold`, `pref_spicy ↔ pref_bland`, `pref_warm_drink ↔ pref_cold_drink`

**Raw vs Context:** Raw = chỉ câu hiện tại (IntentTracker); Context = sau DST, dùng cho **Layer 2 gợi ý** và **export tag** mặc định.

**Output:** `context_scores` (clamp [0,1]) → Layer 2 & Hebbian feedback. State lưu ở `session_state.json` — nên `make layer1-reset-state` khi test phiên mới.

---

## Slide 8 — Dữ liệu & huấn luyện Layer 1 *(Hoàng)*

**Dataset:**
- 5 phiên bản: `dataset_v001` → `dataset_v005`, mỗi epoch ~160 mẫu (`intent_samples.csv`)
- Train trên `active_dataset` hoặc `--all-datasets`; có gộp thêm `intent_train_data_rl.json` (RL offline)

**Cấu hình** (`dl_config.json`): 8 epochs, batch 32, lr 0.001, threshold 0.3
**Artifacts:** `intent_model.pt`, `vocab.json`, `intent_model_meta.json`
**Đánh giá:** micro/macro F1 trên tập validation (20%)

**Lệnh:** `make layer1-train-active` | `make layer1-chat`

---

# === PHÂN MỤC: LAYER 2 — Bùi Đức Anh ===

## Slide 9 — [Section] Layer 2: Adaptive Recommendation *(Đức Anh)*

Slide phân mục mở đầu phần Layer 2.

---

## Slide 10 — Layer 2: Ý tưởng & logic thực hiện (hình minh họa) *(Đức Anh)*

> **Hình:** `docs/images/layer2_logic.png`

**Ý tưởng:**
- Mỗi món ăn là một **vector trọng số trên ~53 tag**; gợi ý = so khớp ngữ cảnh người dùng với vector món
- Hệ thống **tự thích ứng**: trọng số được tinh chỉnh online theo lựa chọn thực tế

**Logic xếp hạng (`recommend`):**
1. Mở rộng `context_scores` qua bảng similarity tag (`layer2_config.json`)
2. **Linear Weight Scoring:** `score(món) = Σ_t activation(tag_t) · weight(món, tag_t)`
3. Sắp xếp theo `(score, popularity, id)` → lấy **Top-K**

**Nguồn dữ liệu:** 100 món (`dishes_100.json` / `food_weight_matrix.json`), runtime tách ở `dataset_v001_dishes_runtime.json`.

---

## Slide 11 — Layer 2: Hebbian Online Learning (logic cập nhật) *(Đức Anh)*

**Cơ chế học sau khi người dùng chọn món:**

- **Reward (món được chọn)** — `apply_context_to_dish`:
  - Tag active (`activation ≥ 0.25`): `weight += lr_positive(0.08) · activation`
  - Tag không active: `weight -= penalty · (0.25 − activation)`
- **Penalty (món trong Top-K nhưng không được chọn)** — `apply_negative_feedback_to_dish`:
  - `weight -= lr_negative(0.02) · activation`
- Trọng số luôn được **clamp về [−1, 1]**, sau đó `save()` xuống runtime matrix

**Kết quả demo:** hiển thị `score_before`, `score_after`, `delta` để thấy hệ thống đã "học".

---

## Slide 12 — Quy ước dữ liệu & vận hành Layer 2 *(Đức Anh)*

- Mỗi layer có `datasets.json` khai báo `active_dataset`; data train tách khỏi runtime
- Runtime L2: `dishes_runtime.json`, `feedback_reports.jsonl`
- Tiện ích: `make layer2-migrate` (canonical), `layer2-check` (schema drift), `layer2-reset-runtime`, `layer2-test`

---

# === PHÂN MỤC: LAYER 3 — Trần Bình Minh ===

## Slide 13 — [Section] Layer 3: Genetic Response *(Minh)*

Slide phân mục mở đầu phần Layer 3.

---

## Slide 14 — Layer 3: Ý tưởng & logic thực hiện (hình minh họa) *(Minh)*

> **Hình:** `docs/images/layer3_logic.png`

**Ý tưởng:**
- Câu trả lời bot được "tiến hóa" từ kho gene câu thoại (opening + action + closing) theo **mood** của người dùng
- Câu thoại nào khiến người dùng chốt món → fitness cao → được ưu tiên dần

**Logic sinh câu (`generate`):**
1. Suy ra `mood_key` từ `context_scores` (mood mạnh nhất, ngưỡng ≥ 0.2)
2. Khởi tạo **quần thể 8 chromosome** (bộ 3 gene) từ pool theo mood
3. Chọn cá thể: **ε-greedy (ε=0.2)** ngẫu nhiên, hoặc **roulette wheel** theo fitness
4. **Mutation:** thêm slang ngẫu nhiên (mutation_rate)
5. Ghép `opening + action + closing` → câu trả lời, chèn tên món vào `{foods}`

---

## Slide 15 — Layer 3: Fitness Update (logic phản hồi ngầm) *(Minh)*

**Cập nhật fitness theo implicit feedback** (`update_fitness`):
- Người dùng **chọn món** → `fitness += 0.25` (củng cố chromosome tốt)
- Người dùng **bỏ qua / thoát** → `fitness -= 0.2` (tối thiểu 0.05 để không triệt tiêu)

**Lưu trữ:** ghi `history` + `runtime_fitness` riêng vào `fitness_history.json` (không phá schema canonical).

**Cơ chế chọn lọc:** fitness càng cao → xác suất được roulette chọn càng lớn ở các lượt sau → câu thoại tự tối ưu theo thời gian.

---

## Slide 16 — Luồng runtime end-to-end *(Hoàng)*

**Kịch bản demo** (`main.py`):
1. Chat 1: "Tối nay trời lạnh, tôi muốn món nước nóng"
2. Chat 2: "Quay xe, giờ tôi muốn món nhanh và tiện lợi" *(DST xử lý đổi ý)*
3. Bot trả: câu trả lời (L3) + Top-5 gợi ý (L2)
4. User chọn món → Hebbian (L2) + Fitness (L3); không chọn → chỉ Fitness (failure)

**Entry:** `process_turn()` → `apply_feedback()` / `apply_abandon_feedback()`

---

## Slide 17 — Reinforcement Learning cho Layer 1 (offline) *(Hoàng)*

**Mục tiêu:** thu thập feedback chọn món để huấn luyện tăng cường Layer 1 — **không train realtime**.

| Bước | Mô tả | Output |
|---|---|---|
| 1. Logging | Ghi event khi user chọn món | `selected_dish_events.jsonl` |
| 2. Simulator | Sinh dữ liệu giả lập | `selected_dish_events_simulated.jsonl` |
| 3. Offline train | Chuyển feedback → train samples | `intent_train_data_rl.json` |

**Thống kê:** 102 events → 102 train samples (`stats.json`)
**Lệnh:** `make layer1-rl-generate` → `layer1-rl-train` → `layer1-rl-check`

---

# === PHÂN MỤC: ĐÁNH GIÁ HIỆU QUẢ — Cả nhóm ===

## Slide 18 — [Section] Đánh giá hiệu quả *(Cả nhóm)*

Slide phân mục mở đầu phần đánh giá định lượng.

---

## Slide 19 — Pipeline đánh giá & phương pháp *(Hoàng)*

**Package:** `src/evaluation/` — `metrics.py`, `layer1_eval.py`, `layer2_eval.py`, `layer3_eval.py`, `pipeline_eval.py`

**Lệnh chạy:**
```bash
make layer3-simulate   # bổ sung data giả lập L3
make eval-run          # đánh giá toàn bộ (skip train L1)
make eval-train-and-run  # train ablation L1 + đánh giá
```

**Tập dữ liệu đánh giá:**
| Layer | Nguồn | Số mẫu |
|---|---|---|
| L1 | Validation 20% intent (all datasets) | 142 |
| L2 Oracle | `intent_samples.csv` tag lý tưởng | 709 |
| L2 Behavioral | RL events (4 runtime + 100 simulated) | 104 |
| L3 | `fitness_history.json` (11 runtime + 50 simulated) | 61 |
| Pipeline E2E | RL events → L2 recommend vs món chọn | 104 |

**DST runtime** (ghi trong manifest): decay=0.55, alpha=0.88, beta=0.4 — ưu tiên raw tag, session yếu hơn.

**Metric chính:** F1 (L1), Hit@K / MRR / NDCG (L2), success rate / fitness (L3), feedback delta (L2 online).

**Lưu ý:** RL events đã log dùng `context_tags` cũ; metric behavioral phản ánh data lịch sử. Phiên chat mới sau chỉnh DST sẽ có context khác.

---

## Slide 20 — Kết quả tổng hợp (hình minh họa) *(Cả nhóm)*

> **Hình:** `docs/images/evaluation_metrics.png`

**Bảng tổng hợp nhanh** (run `20260610_011454`):

| Thành phần | Chỉ số chính | Kết quả |
|---|---|---|
| DST runtime | decay / alpha / beta | **0.55 / 0.88 / 0.4** |
| L1 Intent | Macro F1 | **0.189** |
| L2 Oracle | Hit@5 / MRR | **1.0 / 1.0** |
| L2 Behavioral | Hit@5 / MRR | **0.048 / 0.021** |
| L3 Genetic | Success rate (62 lượt) | **53.2%** |
| Pipeline E2E | Hit@5 | **0.048** |
| Học online L2 | Feedback delta mean | **+0.599** |

**Nhận xét:** Oracle L2 cao → scoring tốt khi tag chuẩn; behavioral thấp → context runtime khó hơn nhiều.

---

## Slide 21 — Chi tiết metric từng layer *(Cả nhóm)*

**Layer 1** (val 142): Micro F1 = 0.291, Macro F1 = 0.189, Precision = 0.748, Recall = 0.181. Ablation RL: delta = 0 (chưa train ablation thành công).

**Layer 2 Oracle** (709): Hit@5 = 1.0, NDCG@5 = 0.971 — upper bound scoring.

**Layer 2 Behavioral** (105): Hit@5 = 0.048, MRR = 0.021, mean rank ≈ 5.83.

**Layer 3** (62 lượt): Success rate tổng 53.2% (runtime 58.3%, simulated 52.0%), 34 chromosome unique, avg fitness 2.57.

**Pipeline E2E:** Tag overlap 0.251; feedback delta +0.599 sau Hebbian update.

**Hạn chế khi báo cáo:** L2/L3 chủ yếu simulated; L3 không có BLEU/ROUGE; behavioral Hit@5 thấp do `context_tags` log cũ (trước tinh chỉnh DST decay=0.55).

Chi tiết: `docs/evaluation_metrics.md`

---

## Slide 22 — Triển khai & DevOps *(Đức Anh)*

- **Makefile:** `make setup`, `app-run`, `app-demo`, `eval-run`, `layer{1,2,3}-*`
- **Docker:** compose riêng từng layer + `docker/app/`
- **Testing:** `test_layer2.py`, `test_layer2_integration.py`, `test_feedback_logging.py`, `test_metrics.py`

---

## Slide 23 — Demo minh họa *(Hoàng)*

```bash
make app-demo
# Chat1: "Tối nay trời lạnh, tôi muốn món nước nóng."
# Bot: [câu trả lời genetic] + Top 5 món + score
# Chat2: "Quay xe, giờ tôi muốn món nhanh và tiện lợi."
# Feedback: chosen=dish_001 → delta score
```

**Điểm nhấn:** DST xử lý "quay xe" · score món đổi sau feedback · câu trả lời khác nhau theo mood.

---

## Slide 24 — Hạn chế & Kết luận *(Cả nhóm)*

**Ưu điểm:** modular, offline, kết hợp DL multi-label + DST + Hebbian + GA + RL offline; **đã có pipeline đánh giá định lượng** (`make eval-run`).

**Kết quả nổi bật:** L2 Oracle Hit@5 = 1.0 (scoring tốt); L3 success rate ≈ 52%; Hebbian feedback delta +0.647.

**Hạn chế:** dataset nhỏ; L2 behavioral Hit@5 thấp (0.039); L1 recall thấp (0.18); L3/L2 eval chủ yếu simulated; chưa có UI.

**Hướng phát triển:** thu thập phiên chat thật, mở rộng corpus, UI chat, ablation RL L1 trên môi trường train ổn định.

**Kết luận:** Food Moo Duu là pipeline NLP end-to-end cho domain ẩm thực tiếng Việt, tự học từ lựa chọn người dùng mà không cần label thủ công. — *Cảm ơn & Q&A*

---

## Phụ lục (không tính trong slide chính)

1. Bảng đầy đủ 53 tag + nhãn tiếng Việt (`data/layer1/tags.json` → `system_tags`).
2. Danh sách 15 conflict pairs (`data/layer1/conflict_pairs.json`).
3. Cấu trúc JSON event RL feedback (schema trong README).
4. Cách dựng lại hình minh họa: `python scripts/build_diagrams.py`.
5. Cách dựng lại slide: `python scripts/build_slides.py`.
6. Chi tiết metric: `docs/evaluation_metrics.md` | run mới nhất: `data/evaluation/runs/run_20260610_011454/`.
7. DST config: `data/common_config.json` (decay=0.55, alpha=0.88, beta=0.4).
