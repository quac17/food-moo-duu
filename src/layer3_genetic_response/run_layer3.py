from src.layer3_genetic_response.genetic_generator import GeneticGenerator


def main() -> None:
    generator = GeneticGenerator()
    generated = generator.generate()
    print("Layer3 ok. Response:", generated["response"])
    generator.update_fitness(chromosome_key=generated["chromosome_key"], success=True)
    print("Layer3 da cap nhat fitness thanh cong.")


if __name__ == "__main__":
    main()
