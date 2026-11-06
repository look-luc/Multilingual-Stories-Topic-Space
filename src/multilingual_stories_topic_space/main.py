from src.data.get_data import get_all_data, get_some_data
from src.model.model import (
    DeBerta,
    Qwen,
    XLM_RoBerta,
    evaluate_cross_lingual_similarity,
    visualize_all_models_single_plot,
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
    data_dict = setup("some")

    print("Running individual model visualizations...")
    for lang, texts in data_dict.items():
        if not texts:
            continue
        deberta.visualize_embeddings(texts, lang, f"deberta_{lang.lower()}_space")
        qwen.visualize_embeddings(texts, lang, f"qwen_{lang.lower()}_space")
        roberta.visualize_embeddings(texts, lang, f"xlmr_{lang.lower()}_space")

    print("\nEvaluating Cross-Lingual Similarities...")
    langs = list(data_dict.keys())
    if len(langs) >= 2:
        lang1, lang2 = langs[0], langs[1]
        texts1 = data_dict[lang1]
        texts2 = data_dict[lang2]

        print(f"Comparing cross-lingual similarity between '{lang1}' and '{lang2}':")
        for model_name, model_instance in models.items():
            print(f"\n--- Model: {model_name} ---")
            emb_lang1 = model_instance.get_embeddings(texts1)
            emb_lang2 = model_instance.get_embeddings(texts2)

            # Use evaluate_cross_lingual_similarity
            similarity_matrix = evaluate_cross_lingual_similarity(emb_lang1, emb_lang2)
            print(f"Similarity Matrix Shape: {similarity_matrix.shape}")

    print("\nGenerating combined model comparative scatterplots...")
    for lang, texts in data_dict.items():
        if not texts:
            continue
        visualize_all_models_single_plot(texts, lang, models, f"combined_{lang.lower()}_space")

    print("Done!")


if __name__ == "__main__":
    main()
