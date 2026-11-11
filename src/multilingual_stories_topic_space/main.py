import re
from itertools import combinations

import numpy as np
import pandas as pd

from src.data.get_data import get_all_data, get_some_data
from src.model.LDA import LDA
from src.model.model import (
    DeBerta,
    Qwen,
    XLM_RoBerta,
    evaluate_cross_lingual_similarity,
    visualize_multilingual_models_single_plot,
)


def parse_silhouette_scores(log_file_path):
    # Regex pattern to match lines like: [Acholi] DeBERTa Silhouette Score: 0.1994
    pattern = re.compile(r"\[(.*?)\]\s+(.*?)\s+Silhouette\s+Score:\s+([\d\.]+)")
    data = []

    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                lang, model, score = match.groups()
                data.append({
                    "Language": lang,
                    "Model": model,
                    "Silhouette Score": float(score)
                })

    if not data:
        print("No silhouette scores found in the specified log file.")
        return None

    df = pd.DataFrame(data)

    # Pivot the table so models are columns and languages are rows
    pivot_df = df.pivot(index="Language", columns="Model", values="Silhouette Score").reset_index()

    # Print Markdown table to console
    print("\n### Silhouette Scores Table\n")
    print(pivot_df.to_markdown(index=False))

    # Export to CSV
    output_csv = "silhouette_scores_table.csv"
    pivot_df.to_csv(output_csv, index=False)
    print(f"\nSaved table to {output_csv}")

    return pivot_df

def setup(data_amount: str = "some"):
    if data_amount.lower() == "some":
        return get_some_data()
    elif data_amount.lower() == "all":
        return get_all_data()
    else:
        raise Exception("Either: 'some' or 'all'")


def main(just_graph:bool=False, lda:bool=False):
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

    if lda:
        deberta_assignment = []
        qwen_assignment = []
        roberta_assignment = []
        print("Training language-specific LDA models and categorizing stories...")
        for lang, texts in data_dict.items():
            if not texts:
                continue
            print(f"\n--- LDA Topics & Categorization for {lang} ---")
            try:
                lda_instance = LDA(docs=texts, num_topics=5)

                # Print topics cleanly
                for idx, topic_desc in lda_instance.lda_model.print_topics(-1):
                    print(f"  Topic {idx}: {topic_desc}")

                # Run classification for each model using the correct instance, texts, and corpus
                print(f"  Running DeBERTa classification for {lang}...")
                deberta_assignments = deberta.lda_category(data=texts, lda_model=lda_instance.lda_model, corpus=lda_instance.corpus)
                deberta_assignment.append(deberta_assignments)

                print(f"  Running Qwen classification for {lang}...")
                qwen_assignments = qwen.lda_category(data=texts, lda_model=lda_instance.lda_model, corpus=lda_instance.corpus)
                qwen_assignment.append(qwen_assignments)

                print(f"  Running XLM-RoBERTa classification for {lang}...")
                roberta_assignments = roberta.lda_category(data=texts, lda_model=lda_instance.lda_model, corpus=lda_instance.corpus)
                roberta_assignment.append(roberta_assignments)

            except Exception as e:
                print(f"  Could not train/categorize LDA for {lang}: {e}")
        print(
            f"""
            deberta assignments: {deberta_assignment}\n
            xlm RoBerta assignments: {roberta_assignment}\n
            Qwen assignments: {qwen_assignment}\n
            """
        )

    if not just_graph:
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
    else:
        visualize_multilingual_models_single_plot(data_dict, models)

    print("Done!")


if __name__ == "__main__":
    parse_silhouette_scores("/Users/lucdenardi/Desktop/code_lang/python/Multilingual-Stories-Topic-Space/35466656.log")
