"""Sinh cac hinh minh hoa (PNG) cho slide bao cao Food Moo Duu.

Cac hinh:
  - system_architecture.png : Kien truc tong the 3 layer + feedback loop (bat buoc)
  - layer1_logic.png        : Logic Layer 1 (Encoder -> heads + DST)
  - layer2_logic.png        : Logic Layer 2 (Linear scoring + Hebbian update)
  - layer3_logic.png        : Logic Layer 3 (Genetic Algorithm loop)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "docs" / "images"

# Bang mau dong bo voi slide
NAVY = "#1F2D3D"
ORANGE = "#E86A17"
TEAL = "#127C8A"
LIGHT = "#F5F1E8"
GREY = "#555B66"
WHITE = "#FFFFFF"
GREEN = "#3A8C5A"
PURPLE = "#6A4C93"

plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text, fc, tc=WHITE, fs=11, bold=True, rounded=0.04):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounded}",
        linewidth=1.5, edgecolor=NAVY, facecolor=fc, mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, color=tc,
        weight="bold" if bold else "normal", wrap=True, zorder=5,
    )


def arrow(ax, x1, y1, x2, y2, color=NAVY, style="-|>", lw=2, rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=18,
            linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=3,
        )
    )


def new_canvas(w=14, h=8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    return fig, ax


def save(fig, name):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / name
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  - {out}")


def diagram_architecture():
    fig, ax = new_canvas(14, 8)
    ax.text(50, 96, "Kiến trúc hệ thống Food Moo Duu", ha="center",
            fontsize=20, weight="bold", color=NAVY)
    ax.text(50, 90.5, "Pipeline 3 tầng offline + vòng lặp học từ phản hồi người dùng",
            ha="center", fontsize=12, color=GREY, style="italic")

    # User input
    box(ax, 5, 74, 22, 9, "Người dùng (chat)\nngôn ngữ tự nhiên", ORANGE, fs=12)

    # 3 layers
    box(ax, 36, 73, 28, 11,
        "LAYER 1\nIntent + Dialog State Tracking\n(vi-SBERT, multi-label, DST)", TEAL, fs=11)
    box(ax, 36, 50, 28, 11,
        "LAYER 2\nAdaptive Recommendation\n(Linear scoring + Hebbian)", TEAL, fs=11)
    box(ax, 36, 27, 28, 11,
        "LAYER 3\nGenetic Response\n(GA + epsilon-greedy + fitness)", TEAL, fs=11)

    # Output to user
    box(ax, 72, 50, 23, 11, "Phản hồi bot\n+ Top-K món gợi ý", ORANGE, fs=11)
    box(ax, 72, 27, 23, 11, "Người dùng\nchọn / bỏ qua món", ORANGE, fs=11)

    # Pipeline arrows down
    arrow(ax, 27, 78.5, 36, 78.5, color=NAVY)
    arrow(ax, 50, 73, 50, 61, color=NAVY)
    ax.text(52, 67, "context_scores", fontsize=9, color=GREY)
    arrow(ax, 50, 50, 50, 38, color=NAVY)
    ax.text(52, 44, "top-K + mood", fontsize=9, color=GREY)

    # Layer2 -> output, Layer3 -> output
    arrow(ax, 64, 55.5, 72, 55.5, color=NAVY)
    arrow(ax, 64, 32.5, 72, 38, color=NAVY, rad=-0.2)

    # User choice
    arrow(ax, 83.5, 50, 83.5, 38, color=ORANGE)

    # Feedback loops (dashed)
    arrow(ax, 72, 31, 64, 53, color=GREEN, ls="--", rad=0.25)
    ax.text(66, 20, "Hebbian update (L2)", fontsize=9, color=GREEN, weight="bold")
    arrow(ax, 78, 27, 55, 27, color=PURPLE, ls="--", rad=0.3)
    ax.text(40, 18, "Fitness update (L3)", fontsize=9, color=PURPLE, weight="bold")

    # RL offline loop note
    box(ax, 5, 50, 24, 11,
        "RL offline (Layer 1)\nlog feedback -> train\ntăng cường định kỳ", GREY, fs=9, rounded=0.06)
    arrow(ax, 72, 28, 29, 55, color=GREY, ls=":", rad=0.15, lw=1.5)

    save(fig, "system_architecture.png")


def diagram_layer1():
    fig, ax = new_canvas(14, 7.5)
    ax.text(50, 95, "Layer 1 - Intent Recognition & Dialog State Tracking",
            ha="center", fontsize=17, weight="bold", color=NAVY)

    box(ax, 3, 64, 17, 12, "Câu tiếng Việt\n(input)", ORANGE, fs=11)
    box(ax, 23, 64, 18, 12, "Chuẩn hóa VN\n+ tokenize", GREY, fs=10)
    box(ax, 44, 64, 20, 12, "Encoder\nvi-SBERT\n(mean pooling)", TEAL, fs=10)

    # Two heads
    box(ax, 70, 78, 27, 9, "Projection head\n(metric learning)", PURPLE, fs=10)
    box(ax, 70, 63, 27, 9, "Classification head\n(sigmoid -> tag scores)", TEAL, fs=10)

    arrow(ax, 20, 70, 23, 70)
    arrow(ax, 41, 70, 44, 70)
    arrow(ax, 64, 70, 70, 82.5, rad=-0.2)
    arrow(ax, 64, 70, 70, 67.5, rad=0.1)

    # Loss note
    box(ax, 70, 47, 27, 10,
        "Loss = BCE (multi-label)\n+ alpha * metric loss\n(overlap-based similarity)", LIGHT, tc=NAVY, fs=9)
    arrow(ax, 83.5, 78, 83.5, 57, color=GREY, ls="--")
    arrow(ax, 83.5, 63, 83.5, 57, color=GREY, ls="--")

    # DST block
    box(ax, 8, 30, 84, 9, "DIALOG STATE TRACKER (DST) - tích lũy ngữ cảnh qua nhiều lượt",
        NAVY, fs=12)
    arrow(ax, 70, 65, 50, 39, color=NAVY, rad=0.2)
    ax.text(58, 50, "raw_scores", fontsize=9, color=GREY)

    box(ax, 6, 12, 27, 12,
        "1. Decay\nscore *= 0.92\n(hao mòn theo lượt)", GREEN, fs=9)
    box(ax, 37, 12, 27, 12,
        "2. Accumulation\nscore += 0.65 * conf\n(cộng dồn tín hiệu mới)", TEAL, fs=9)
    box(ax, 68, 12, 27, 12,
        "3. Conflict resolution\ngiảm tag yếu theo gap\n(vd: hot vs cold)", ORANGE, fs=9)

    arrow(ax, 33, 33, 19, 24, color=NAVY)
    arrow(ax, 50, 30, 50, 24, color=NAVY)
    arrow(ax, 67, 33, 81, 24, color=NAVY)

    box(ax, 35, 1.5, 30, 7, "context_scores -> Layer 2", ORANGE, fs=10)
    arrow(ax, 50, 12, 50, 8.5, color=NAVY)

    save(fig, "layer1_logic.png")


def diagram_layer2():
    fig, ax = new_canvas(14, 7.5)
    ax.text(50, 95, "Layer 2 - Adaptive Recommendation (Linear Scoring + Hebbian)",
            ha="center", fontsize=16, weight="bold", color=NAVY)

    box(ax, 4, 70, 22, 12, "context_scores\n(từ Layer 1)", ORANGE, fs=11)
    box(ax, 4, 50, 22, 13, "Ma trận trọng số\ntag x món\n(100 món, ~53 tag)", NAVY, fs=10)

    box(ax, 36, 60, 28, 14,
        "Linear Weight Scoring\nscore(món) = SUM\nactivation(tag) * weight(món,tag)",
        TEAL, fs=10)

    box(ax, 73, 60, 23, 14, "Xếp hạng\n-> Top-K món\ngợi ý", ORANGE, fs=11)

    arrow(ax, 26, 76, 36, 70)
    arrow(ax, 26, 56, 36, 63, rad=-0.1)
    arrow(ax, 64, 67, 73, 67)

    # Feedback
    box(ax, 73, 38, 23, 11, "Người dùng chọn\n/ bỏ qua món", PURPLE, fs=10)
    arrow(ax, 84.5, 60, 84.5, 49, color=NAVY)

    box(ax, 8, 22, 40, 13,
        "HEBBIAN (+): món được chọn\nweight += lr_pos(0.08) * activation\n(tag active >= 0.25)",
        GREEN, fs=9)
    box(ax, 54, 22, 40, 13,
        "PENALTY (-): món trong top-K\nkhông được chọn\nweight -= lr_neg(0.02) * activation",
        ORANGE, fs=9)

    arrow(ax, 80, 38, 35, 35, color=GREEN, ls="--", rad=0.15)
    arrow(ax, 84, 38, 74, 35, color=ORANGE, ls="--", rad=-0.1)

    box(ax, 30, 5, 40, 9,
        "Cập nhật runtime matrix -> hệ thống\ntự học theo thời gian", LIGHT, tc=NAVY, fs=10)
    arrow(ax, 28, 22, 45, 14, color=NAVY)
    arrow(ax, 74, 22, 55, 14, color=NAVY)

    save(fig, "layer2_logic.png")


def diagram_layer3():
    fig, ax = new_canvas(14, 7.5)
    ax.text(50, 95, "Layer 3 - Genetic Response Generation",
            ha="center", fontsize=17, weight="bold", color=NAVY)

    box(ax, 4, 72, 22, 12, "context_scores\n-> mood_key", ORANGE, fs=10)
    box(ax, 4, 52, 22, 13, "Gene pool\nOpening / Action /\nClosing (theo mood)", NAVY, fs=9)

    box(ax, 35, 70, 28, 13,
        "Khởi tạo quần thể\n8 chromosome\n(bộ 3 gene)", TEAL, fs=10)

    box(ax, 70, 70, 27, 13,
        "Chọn cá thể\nepsilon-greedy (e=0.2)\nhoặc roulette (fitness)", PURPLE, fs=9)

    box(ax, 70, 48, 27, 12, "Mutation\n(thêm slang ngẫu nhiên)", GREEN, fs=10)
    box(ax, 35, 48, 28, 12, "Ghép câu thoại\nopening+action+closing", TEAL, fs=10)

    box(ax, 35, 28, 28, 11, "Câu trả lời bot\n-> người dùng", ORANGE, fs=11)

    arrow(ax, 26, 78, 35, 76.5)
    arrow(ax, 26, 58, 35, 73, rad=-0.15)
    arrow(ax, 63, 76.5, 70, 76.5)
    arrow(ax, 83.5, 70, 83.5, 60)
    arrow(ax, 70, 54, 63, 54)
    arrow(ax, 49, 48, 49, 39)

    # Fitness feedback loop
    box(ax, 8, 10, 38, 12,
        "Chọn món -> fitness += 0.25\n(củng cố câu thoại tốt)", GREEN, fs=9)
    box(ax, 54, 10, 38, 12,
        "Bỏ qua/thoát -> fitness -= 0.2\n(tối thiểu 0.05)", ORANGE, fs=9)

    arrow(ax, 35, 31, 27, 22, color=NAVY)
    arrow(ax, 63, 31, 73, 22, color=NAVY)
    # loop back to selection
    arrow(ax, 27, 22, 83.5, 70, color=GREY, ls="--", rad=-0.3, lw=1.5)
    ax.text(15, 40, "cập nhật\nfitness_map", fontsize=9, color=GREY, weight="bold")

    save(fig, "layer3_logic.png")


def build():
    print("Sinh hình minh họa:")
    diagram_architecture()
    diagram_layer1()
    diagram_layer2()
    diagram_layer3()
    print("Hoàn tất.")


if __name__ == "__main__":
    build()
