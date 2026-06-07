from __future__ import annotations

import argparse
import sys

from src.layer3_genetic_response.genetic_generator import GeneticGenerator


def _safe_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer3 genetic response smoke/demo")
    parser.add_argument("--success", action="store_true", help="Cap nhat fitness theo huong success")
    parser.add_argument("--repeat", type=int, default=1, help="So cau sinh de test")
    args = parser.parse_args()

    generator = GeneticGenerator()
    repeat = max(1, args.repeat)
    last_key = ""
    for idx in range(repeat):
        generated = generator.generate()
        last_key = generated["chromosome_key"]
        print(f"Layer3 response [{idx + 1}/{repeat}]:", _safe_text(generated["response"]))

    if last_key:
        generator.update_fitness(chromosome_key=last_key, success=args.success)
        status = "success" if args.success else "failure"
        print(f"Layer3 da cap nhat fitness thanh cong ({status}).")


if __name__ == "__main__":
    main()
