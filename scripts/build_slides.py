"""Sinh file slide PPTX bao cao du an Food Moo Duu."""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Pt, Emu

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "docs" / "food_moo_duu_slides.pptx"
IMG_DIR = ROOT / "docs" / "images"

# Bang mau
NAVY = RGBColor(0x1F, 0x2D, 0x3D)
ORANGE = RGBColor(0xE8, 0x6A, 0x17)
TEAL = RGBColor(0x12, 0x7C, 0x8A)
LIGHT = RGBColor(0xF5, 0xF1, 0xE8)
GREY = RGBColor(0x55, 0x5B, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

# Kich thuoc slide 16:9
SLIDE_W = Emu(12192000)
SLIDE_H = Emu(6858000)


def add_background(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_box(slide, left, top, width, height):
    box = slide.shapes.add_textbox(Emu(left), Emu(top), Emu(width), Emu(height))
    tf = box.text_frame
    tf.word_wrap = True
    return box, tf


def style_run(run, size, color, bold=False, italic=False, font="Calibri"):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font


def add_accent_bar(slide, color=ORANGE):
    bar = slide.shapes.add_shape(
        1, Emu(0), Emu(0), Emu(160000), SLIDE_H
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()


def title_slide(prs, title, subtitle, members):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, NAVY)
    add_accent_bar(slide, ORANGE)

    _, tf = add_box(slide, 900000, 1700000, 10400000, 1800000)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    style_run(r, 44, WHITE, bold=True)

    p2 = tf.add_paragraph()
    p2.space_before = Pt(10)
    r2 = p2.add_run()
    r2.text = subtitle
    style_run(r2, 20, ORANGE, italic=True)

    _, tf3 = add_box(slide, 900000, 4100000, 10400000, 1800000)
    head = tf3.paragraphs[0]
    rh = head.add_run()
    rh.text = "Nhóm thực hiện"
    style_run(rh, 16, TEAL, bold=True)
    for name, role in members:
        pm = tf3.add_paragraph()
        rm = pm.add_run()
        rm.text = f"{name}  -  {role}"
        style_run(rm, 16, LIGHT)
    return slide


def content_slide(prs, title, blocks, owner=None):
    """blocks: list cua (loai, noi_dung).
    loai: 'h' (heading), 'b' (bullet cap 1), 's' (bullet cap 2), 'note'."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT)
    add_accent_bar(slide, ORANGE)

    # Tieu de
    _, tf_title = add_box(slide, 500000, 350000, 9800000, 900000)
    p = tf_title.paragraphs[0]
    r = p.add_run()
    r.text = title
    style_run(r, 30, NAVY, bold=True)

    # Owner badge
    if owner:
        badge = slide.shapes.add_shape(1, Emu(9400000), Emu(380000), Emu(2400000), Emu(520000))
        badge.fill.solid()
        badge.fill.fore_color.rgb = TEAL
        badge.line.fill.background()
        btf = badge.text_frame
        btf.word_wrap = True
        bp = btf.paragraphs[0]
        bp.alignment = PP_ALIGN.CENTER
        br = bp.add_run()
        br.text = owner
        style_run(br, 12, WHITE, bold=True)

    # Noi dung
    _, tf = add_box(slide, 600000, 1450000, 11000000, 5000000)
    first = True
    for kind, text in blocks:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        if kind == "h":
            p.space_before = Pt(10)
            r = p.add_run()
            r.text = text
            style_run(r, 19, ORANGE, bold=True)
        elif kind == "b":
            p.space_before = Pt(5)
            r = p.add_run()
            r.text = "•  " + text
            style_run(r, 16, NAVY)
        elif kind == "s":
            p.level = 1
            r = p.add_run()
            r.text = "–  " + text
            style_run(r, 14, GREY)
        elif kind == "note":
            p.space_before = Pt(8)
            r = p.add_run()
            r.text = text
            style_run(r, 13, TEAL, italic=True)
    return slide


def section_slide(prs, title, owner):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, TEAL)
    add_accent_bar(slide, ORANGE)
    _, tf = add_box(slide, 900000, 2600000, 10400000, 1600000)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    style_run(r, 36, WHITE, bold=True)
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = f"Phụ trách: {owner}"
    style_run(r2, 18, LIGHT, italic=True)
    return slide


def add_image_centered(slide, image_name, top, max_w, max_h):
    """Chen anh, can giua ngang, giu ti le, gioi han trong khung max_w x max_h (EMU)."""
    path = IMG_DIR / image_name
    if not path.exists():
        return
    from PIL import Image

    with Image.open(path) as im:
        iw, ih = im.size
    ratio = iw / ih
    width = max_w
    height = int(width / ratio)
    if height > max_h:
        height = max_h
        width = int(height * ratio)
    left = int((SLIDE_W - width) / 2)
    slide.shapes.add_picture(str(path), Emu(left), Emu(top), Emu(width), Emu(height))


def image_slide(prs, title, image_name, blocks, owner=None, img_height=3400000):
    """Slide: tieu de + bullet ngan o tren + hinh minh hoa lon o duoi."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT)
    add_accent_bar(slide, ORANGE)

    _, tf_title = add_box(slide, 500000, 300000, 9800000, 850000)
    p = tf_title.paragraphs[0]
    r = p.add_run()
    r.text = title
    style_run(r, 26, NAVY, bold=True)

    if owner:
        badge = slide.shapes.add_shape(1, Emu(9400000), Emu(330000), Emu(2400000), Emu(520000))
        badge.fill.solid()
        badge.fill.fore_color.rgb = TEAL
        badge.line.fill.background()
        btf = badge.text_frame
        bp = btf.paragraphs[0]
        bp.alignment = PP_ALIGN.CENTER
        br = bp.add_run()
        br.text = owner
        style_run(br, 12, WHITE, bold=True)

    # Bullet ngan
    _, tf = add_box(slide, 600000, 1250000, 11000000, 1900000)
    first = True
    for kind, text in blocks:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        if kind == "h":
            r = p.add_run()
            r.text = text
            style_run(r, 15, ORANGE, bold=True)
        elif kind == "b":
            r = p.add_run()
            r.text = "•  " + text
            style_run(r, 13, NAVY)
        elif kind == "note":
            r = p.add_run()
            r.text = text
            style_run(r, 12, TEAL, italic=True)

    # Hinh minh hoa o duoi
    add_image_centered(slide, image_name, top=3300000, max_w=10600000, max_h=img_height)
    return slide


def build():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    members = [
        ("Trần Việt Hoàng", "Layer 1 - Định nghĩa Tag + Kiến trúc tổng thể"),
        ("Bùi Đức Anh", "Layer 2 - Adaptive Recommendation"),
        ("Trần Bình Minh", "Layer 3 - Genetic Response"),
    ]

    # Slide 1 - Bia
    title_slide(
        prs,
        "Food Moo Duu - Hệ thống gợi ý món ăn thông minh offline",
        '"Là bò, sáng phải rống - Là người, sống phải ráng"',
        members,
    )

    # Slide 2 - Van de & muc tieu
    content_slide(
        prs,
        "Vấn đề & Mục tiêu",
        [
            ("h", "Vấn đề"),
            ("b", "Người dùng mô tả nhu cầu ăn uống bằng ngôn ngữ tự nhiên, đa chiều (thời gian, thời tiết, tâm trạng, khẩu vị)"),
            ("b", "Cần hệ thống offline, không phụ thuộc API cloud"),
            ("b", "Cần học từ phản hồi khi người dùng chọn hoặc bỏ qua gợi ý"),
            ("h", "Mục tiêu"),
            ("b", "Xây dựng chatbot gợi ý món ăn theo kiến trúc 3 tầng"),
            ("b", "Theo dõi ngữ cảnh hội thoại nhiều lượt (multi-turn)"),
            ("b", "Tự cập nhật mô hình gợi ý và câu trả lời theo feedback ngầm"),
        ],
        owner="Hoàng",
    )

    # Slide 3 - Tong quan giai phap
    content_slide(
        prs,
        "Tổng quan giải pháp",
        [
            ("h", "Ý tưởng cốt lõi"),
            ("b", "Pipeline xử lý tuần tự 3 layer: DL + học trực tuyến + thuật toán di truyền"),
            ("h", "Điểm nổi bật"),
            ("b", "Offline end-to-end qua main.py -> FoodSuggestionPipeline"),
            ("b", "Mỗi layer chạy độc lập được (train/test/docker riêng)"),
            ("b", "Dữ liệu train và runtime tách biệt rõ ràng"),
            ("h", "Công nghệ"),
            ("b", "Python, PyTorch, scikit-learn, pandas, Docker, Makefile"),
        ],
        owner="Hoàng",
    )

    # Slide 4 - Kien truc he thong (HINH BAT BUOC)
    image_slide(
        prs,
        "Kiến trúc hệ thống (sơ đồ tổng thể)",
        "system_architecture.png",
        [
            ("h", "Luồng xử lý + 2 vòng phản hồi"),
            ("b", "User -> L1 (Intent+DST) -> L2 (Scoring) -> L3 (Response) -> User"),
            ("b", "Feedback: Hebbian update (L2) + Fitness update (L3); RL offline cho L1"),
        ],
        owner="Hoàng",
    )

    # ===== PHAN MUC LAYER 1 =====
    section_slide(prs, "Layer 1 - Intent & Dialog State Tracking", "Trần Việt Hoàng")

    # Slide 6 - Layer 1: Y tuong & logic (HINH)
    image_slide(
        prs,
        "Layer 1: Ý tưởng & logic thực hiện",
        "layer1_logic.png",
        [
            ("h", "Ý tưởng: biến câu nói -> bộ tag ngữ cảnh (multi-label) + học embedding theo ngữ cảnh"),
            ("b", "Chuẩn hóa VN + tokenize -> Encoder vi-SBERT (mean pooling)"),
            ("b", "2 head: Classification (sigmoid -> ~53 tag) + Projection (metric learning)"),
            ("note", "Loss = BCE multi-label + alpha(0.35) * metric loss (overlap-based similarity)"),
        ],
        owner="Hoàng",
    )

    # Slide 7 - Layer 1: DST logic
    content_slide(
        prs,
        "Layer 1: Dialog State Tracking (logic DST)",
        [
            ("h", "Vai trò: tích lũy ngữ cảnh nhiều lượt, xử lý người dùng đổi ý"),
            ("h", "3 quy luật cập nhật state (update_context)"),
            ("b", "1. Decay: score *= 0.92 mỗi lượt (thông tin cũ hao mòn)"),
            ("b", "2. Accumulation: score += 0.65 * confidence (cộng dồn intent mới)"),
            ("b", "3. Conflict resolution: cặp tag đối nghịch -> giảm bên yếu theo gap (beta=0.35)"),
            ("s", "VD: weather_hot <-> weather_cold, pref_spicy <-> pref_bland"),
            ("note", "Output: context_scores (clamp [0,1]) -> Layer 2 & Layer 3; lưu session_state.json"),
        ],
        owner="Hoàng",
    )

    # Slide 8 - Du lieu & huan luyen Layer 1
    content_slide(
        prs,
        "Dữ liệu & Huấn luyện Layer 1",
        [
            ("h", "Dataset"),
            ("b", "5 phiên bản: dataset_v001 -> dataset_v005, mỗi epoch ~160 mẫu"),
            ("b", "Train trên active_dataset hoặc --all-datasets; gộp thêm RL offline"),
            ("h", "Cấu hình (dl_config.json)"),
            ("b", "8 epochs, batch 32, lr 0.001, threshold tag 0.3"),
            ("b", "Artifacts: intent_model.pt, vocab.json, intent_model_meta.json"),
            ("b", "Đánh giá: micro/macro F1 trên validation (20%)"),
            ("note", "Lệnh: make layer1-train-active | make layer1-chat"),
        ],
        owner="Hoàng",
    )

    # ===== PHAN MUC LAYER 2 =====
    section_slide(prs, "Layer 2 - Adaptive Recommendation", "Bùi Đức Anh")

    # Slide 10 - Layer 2: Y tuong & logic (HINH)
    image_slide(
        prs,
        "Layer 2: Ý tưởng & logic thực hiện",
        "layer2_logic.png",
        [
            ("h", "Ý tưởng: mỗi món = vector trọng số trên ~53 tag; gợi ý = so khớp ngữ cảnh"),
            ("b", "Mở rộng context qua similarity tag (layer2_config.json)"),
            ("b", "Linear scoring: score(món) = SUM activation(tag) * weight(món, tag)"),
            ("note", "Xếp hạng theo (score, popularity, id) -> Top-K; nguồn: 100 món"),
        ],
        owner="Đức Anh",
    )

    # Slide 11 - Layer 2: Hebbian logic
    content_slide(
        prs,
        "Layer 2: Hebbian Online Learning (logic cập nhật)",
        [
            ("h", "Reward - món được chọn (apply_context_to_dish)"),
            ("b", "Tag active (>= 0.25): weight += lr_positive(0.08) * activation"),
            ("b", "Tag không active: weight -= penalty * (0.25 - activation)"),
            ("h", "Penalty - món trong Top-K nhưng không được chọn"),
            ("b", "weight -= lr_negative(0.02) * activation"),
            ("b", "Trọng số luôn clamp [-1, 1] rồi save() xuống runtime matrix"),
            ("note", "Demo: hiển thị score_before, score_after, delta để thấy hệ thống đã học"),
        ],
        owner="Đức Anh",
    )

    # Slide 12 - Quy uoc du lieu & van hanh Layer 2
    content_slide(
        prs,
        "Quy ước dữ liệu & Vận hành Layer 2",
        [
            ("h", "Convention"),
            ("b", "Mỗi layer có datasets.json khai báo active_dataset"),
            ("b", "Data train tách khỏi runtime (dishes_runtime.json, feedback_reports.jsonl)"),
            ("h", "Tiện ích vận hành"),
            ("b", "make layer2-migrate (canonical) | layer2-check (schema drift)"),
            ("b", "make layer2-reset-runtime | layer2-test"),
        ],
        owner="Đức Anh",
    )

    # ===== PHAN MUC LAYER 3 =====
    section_slide(prs, "Layer 3 - Genetic Response", "Trần Bình Minh")

    # Slide 14 - Layer 3: Y tuong & logic (HINH)
    image_slide(
        prs,
        "Layer 3: Ý tưởng & logic thực hiện",
        "layer3_logic.png",
        [
            ("h", "Ý tưởng: câu thoại được 'tiến hóa' từ gene pool theo mood; câu tốt -> fitness cao"),
            ("b", "Suy mood_key từ context_scores -> khởi tạo 8 chromosome (bộ 3 gene)"),
            ("b", "Chọn cá thể: epsilon-greedy (0.2) hoặc roulette theo fitness; mutation slang"),
            ("note", "Ghép opening+action+closing -> câu trả lời, chèn tên món vào {foods}"),
        ],
        owner="Minh",
    )

    # Slide 15 - Layer 3: Fitness update logic
    content_slide(
        prs,
        "Layer 3: Fitness Update (logic phản hồi ngầm)",
        [
            ("h", "Cập nhật fitness theo implicit feedback (update_fitness)"),
            ("b", "Người dùng chọn món -> fitness += 0.25 (củng cố chromosome tốt)"),
            ("b", "Bỏ qua / thoát -> fitness -= 0.2 (tối thiểu 0.05 để không triệt tiêu)"),
            ("h", "Lưu trữ"),
            ("b", "Ghi history + runtime_fitness riêng vào fitness_history.json"),
            ("note", "Fitness cao -> xác suất roulette chọn cao hơn -> câu thoại tự tối ưu theo thời gian"),
        ],
        owner="Minh",
    )

    # Slide 16 - Luong runtime end-to-end
    content_slide(
        prs,
        "Luồng runtime end-to-end",
        [
            ("h", "Kịch bản demo (main.py)"),
            ("b", "Chat lượt 1: 'Tối nay trời lạnh, tôi muốn món nước nóng'"),
            ("b", "Chat lượt 2..5: có thể quay xe đổi ý (DST cập nhật ngữ cảnh)"),
            ("b", "Sau mỗi lượt: hệ thống trả câu trả lời (L3) + top-K gợi ý (L2)"),
            ("b", "User chọn món -> Hebbian (L2) + fitness (L3); không chọn -> chỉ fitness"),
            ("note", "Entry: process_turn() -> apply_feedback() / apply_abandon_feedback()"),
        ],
        owner="Hoàng",
    )

    # Slide 17 - RL Layer 1 offline
    content_slide(
        prs,
        "Reinforcement Learning cho Layer 1 (offline)",
        [
            ("h", "Mục tiêu: thu thập feedback chọn món để train tăng cường - không train realtime"),
            ("b", "1. Logging: ghi event khi user chọn món -> selected_dish_events.jsonl"),
            ("b", "2. Simulator: sinh dữ liệu giả lập -> selected_dish_events_simulated.jsonl"),
            ("b", "3. Offline train: chuyển feedback -> intent_train_data_rl.json"),
            ("b", "Thống kê: 102 events -> 102 train samples (stats.json)"),
            ("note", "Lệnh: make layer1-rl-generate -> layer1-rl-train -> layer1-rl-check"),
        ],
        owner="Hoàng",
    )

    # Slide 18 - DevOps
    content_slide(
        prs,
        "Triển khai & DevOps",
        [
            ("h", "Makefile - một lệnh cho mỗi tác vụ"),
            ("b", "make setup | app-run | app-demo | layer{1,2,3}-*"),
            ("h", "Docker"),
            ("b", "docker/layer1|2|3/docker-compose.yml + docker/app/ (full app)"),
            ("h", "Testing"),
            ("b", "test_layer2.py, test_layer2_integration.py, test_feedback_logging.py"),
        ],
        owner="Đức Anh",
    )

    # Slide 19 - Demo
    content_slide(
        prs,
        "Demo minh họa",
        [
            ("h", "Ví dụ non-interactive (make app-demo)"),
            ("s", "Chat 1: 'Tối nay trời lạnh, tôi muốn món nước nóng'"),
            ("s", "Bot: câu trả lời genetic + Top 5 món + score"),
            ("s", "Chat 2..5: 'Quay xe, giờ tôi muốn món nhanh và tiện lợi'"),
            ("s", "Feedback: chosen=dish_001 -> delta score"),
            ("h", "Điểm cần nhấn"),
            ("b", "DST xử lý 'quay xe' | score món đổi sau feedback | câu trả lời theo mood"),
        ],
        owner="Minh",
    )

    # Slide 20 - Danh gia, han che & ket luan
    content_slide(
        prs,
        "Đánh giá, Hạn chế & Kết luận",
        [
            ("h", "Ưu điểm"),
            ("b", "Modular, offline, kết hợp DL multi-label + DST + Hebbian + GA + RL offline"),
            ("h", "Hạn chế"),
            ("b", "Dataset nhỏ (~160 mẫu/epoch, 100 món); L3 template cố định; chưa có UI"),
            ("h", "Hướng phát triển"),
            ("b", "Mở rộng corpus, UI chat, đánh giá định lượng, RL train định kỳ"),
            ("note", "Food Moo Duu: pipeline NLP end-to-end tự học từ lựa chọn người dùng. Cảm ơn - Q&A"),
        ],
        owner="Cả nhóm",
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_FILE)
    print(f"Đã tạo slide: {OUT_FILE}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
