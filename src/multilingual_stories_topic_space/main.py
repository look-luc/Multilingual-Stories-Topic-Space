from itertools import combinations

import numpy as np

from src.data.get_data import get_all_data, get_some_data
from src.model.model import (
    DeBerta,
    Qwen,
    XLM_RoBerta,
    evaluate_cross_lingual_similarity,
    visualize_multilingual_models_single_plot,
)


def setup(data_amount: str = "some"):
    if data_amount.lower() == "some":
        return get_some_data()
    elif data_amount.lower() == "all":
        return get_all_data()
    else:
        raise Exception("Either: 'some' or 'all'")


def main():
    print("Initializing models...")
    deberta = DeBerta()
    qwen = Qwen()
    roberta = XLM_RoBerta()

    models = {
        "DeBERTa": deberta,
        "Qwen": qwen,
        "XLM-RoBERTa": roberta
    }

    print("Fetching data...")
    data_dict = setup("all")

    print("Running individual model visualizations...")
    for lang, texts in data_dict.items():
        if not texts:
            continue
        deberta.visualize_embeddings(texts, lang, f"deberta_{lang.lower()}_space")
        qwen.visualize_embeddings(texts, lang, f"qwen_{lang.lower()}_space")
        roberta.visualize_embeddings(texts, lang, f"xlmr_{lang.lower()}_space")

    print("\nEvaluating Cross-Lingual Similarities for All Language Pairs...")
    langs = list(data_dict.keys())

    for model_name, model_instance in models.items():
        print(f"\n--- Model: {model_name} ---")
        # Cache model embeddings per language to avoid redundant computations
        lang_embeddings = {
            lang: model_instance.get_embeddings(texts)
            for lang, texts in data_dict.items() if texts
        }

        valid_langs = list(lang_embeddings.keys())
        for lang1, lang2 in combinations(valid_langs, 2):
            similarity_matrix = evaluate_cross_lingual_similarity(
                lang_embeddings[lang1],
                lang_embeddings[lang2]
            )
            print(f"  {lang1} vs {lang2} | Similarity Matrix Shape: {similarity_matrix.shape} | Mean Cosine Similarity: {np.mean(similarity_matrix):.4f}")

    print("\nGenerating combined model comparative scatterplots...")
    visualize_multilingual_models_single_plot(data_dict, models)

    print("Done!")


if __name__ == "__main__":
    main()
