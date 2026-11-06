from src.data.get_data import get_all_data, get_some_data
from src.model.model import (
    DeBerta,
    Qwen,
    XLM_RoBerta,
    compute_topic_coherence,
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
    qwen_model = Qwen()
    roberta = XLM_RoBerta()
    de_berta = DeBerta()

    print("Fetching data...")
    data_dict = setup("all")

    print("Running visualizations and evaluations...")
    for lang, texts in data_dict.items():
        if not texts:
            print(f"Skipping {lang}: No text data retrieved.")
            continue

        print(f"Processing {lang}...")

        # Run LDA model visualization
        lda_model.visualize_embeddings(
            data=texts,
            lang=lang,
            output_name=f"lda_{lang.lower()}_space"
        )

        # Run XLM-RoBERTa model visualization
        roberta.visualize_embeddings(
            data=texts,
            lang=lang,
            output_name=f"xlmr_{lang.lower()}_space"
        )

        models = {
            "DeBERTa": deberta,
            "Qwen": qwen,
            "XLM-RoBERTa": roberta
        }

    for lang, texts in data_dict.items():
        if not texts:
            continue

        print(f"Generating combined scatterplot for {lang}...")
        visualize_all_models_single_plot(
            data=texts,
            lang=lang,
            models_dict=models,
            output_name=f"combined_{lang.lower()}_space"
        )

    print("Done! Check your directory for output PNG files.")

if __name__ == "__main__":
    main()
