# Khung Slide Báo Cáo — Food Moo Duu

> Hệ thống gợi ý món ăn thông minh offline theo kiến trúc 3 layer (NLP tiếng Việt).
> Khung ~20 slide, soạn dựa trên cấu trúc code, README và luồng runtime thực tế của repo.

**Phân công nhóm:**
- **Trần Việt Hoàng** — Layer 1 (Tag Define) + Kiến trúc tổng thể
- **Bùi Đức Anh** — Layer 2 (Adaptive Recommendation)
- **Trần Bình Minh** — Layer 3 (Genetic Response)

**Hình minh họa** (sinh bằng `scripts/build_diagrams.py`, lưu tại `docs/images/`):
- `system_architecture.png` — kiến trúc tổng thể 3 layer + feedback loop
- `layer1_logic.png` — logic Layer 1 (Encoder → heads + DST)
- `layer2_logic.png` — logic Layer 2 (Linear scoring + Hebbian)
- `layer3_logic.png` — logic Layer 3 (Genetic Algorithm)

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

**3 quy luật cập nhật state** (`DialogStateTracker.update_context`):
1. **Decay:** `score *= 0.92` mỗi lượt — thông tin cũ hao mòn dần
2. **Accumulation:** `score += 0.65 · confidence` — cộng dồn tín hiệu intent mới
3. **Conflict resolution:** với cặp tag đối nghịch, giảm bên yếu theo gap (`β=0.35`)
   - VD conflict pairs: `weather_hot ↔ weather_cold`, `pref_spicy ↔ pref_bland`, `pref_warm_drink ↔ pref_cold_drink`

**Output:** `context_scores` (đã clamp [0,1]) → đầu vào cho Layer 2 & Layer 3. State lưu ở `session_state.json`.

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

## Slide 18 — Triển khai & DevOps *(Đức Anh)*

- **Makefile:** `make setup`, `app-run`, `app-demo`, `layer{1,2,3}-*`
- **Docker:** compose riêng từng layer + `docker/app/`
- **Testing:** `test_layer2.py`, `test_layer2_integration.py`, `test_feedback_logging.py`

---

## Slide 19 — Demo minh họa *(Minh)*

```bash
make app-demo
# Chat1: "Tối nay trời lạnh, tôi muốn món nước nóng."
# Bot: [câu trả lời genetic] + Top 5 món + score
# Chat2: "Quay xe, giờ tôi muốn món nhanh và tiện lợi."
# Feedback: chosen=dish_001 → delta score
```

**Điểm nhấn:** DST xử lý "quay xe" · score món đổi sau feedback · câu trả lời khác nhau theo mood.

---

## Slide 20 — Đánh giá, hạn chế & Kết luận *(Cả nhóm)*

**Ưu điểm:** modular, offline, kết hợp nhiều kỹ thuật NLP (DL multi-label, DST, Hebbian, GA), có RL offline.

**Hạn chế:** dataset nhỏ (~160 mẫu/epoch, 100 món); Layer 3 dùng template cố định; chưa có UI; chưa báo cáo metric tập trung.

**Hướng phát triển:** mở rộng corpus, UI chat, đánh giá định lượng, tích hợp RL train định kỳ.

**Kết luận:** Food Moo Duu là pipeline NLP end-to-end cho domain ẩm thực tiếng Việt, tự học từ lựa chọn người dùng mà không cần label thủ công. — *Cảm ơn & Q&A*

---

## Phụ lục (không tính trong slide chính)

1. Bảng đầy đủ 53 tag + nhãn tiếng Việt (`data/layer1/tags.json` → `system_tags`).
2. Danh sách 15 conflict pairs (`data/layer1/conflict_pairs.json`).
3. Cấu trúc JSON event RL feedback (schema trong README).
4. Cách dựng lại hình minh họa: `python scripts/build_diagrams.py`.
5. Cách dựng lại slide: `python scripts/build_slides.py`.
