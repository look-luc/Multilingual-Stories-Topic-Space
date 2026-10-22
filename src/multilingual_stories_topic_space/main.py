import src.model.model as get_model
from src.data.get_data import get_some_data


def setup(data_amount: str = "some"):
    if data_amount.lower() == "some":
        return get_some_data()
    elif data_amount.lower() == "all":
        raise Exception("TBD")
    else:
        raise Exception("Either: 'some' or 'all'")

def main():
    print("Initializing models...")
    lda_model = get_model.LDA()
    roberta = get_model.XLM_RoBerta()

    print("Fetching data...")
    data_dict = setup("some")

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

    print("Done! Check your directory for output PNG files.")

if __name__ == "__main__":
    main()
