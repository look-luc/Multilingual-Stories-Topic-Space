import re
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.cross_decomposition import CCA

from src.data.get_data import get_all_data, get_some_data
from src.model.LDA import LDA
from src.model.model import (
    DeBerta,
    Qwen,
    XLM_RoBerta,
    evaluate_cross_lingual_similarity,
    visualize_multilingual_models_single_plot,
)


def calculate_topic_jsd(lda_model, topic_id_1, topic_id_2, num_words=None):
    """
    Computes JSD between the word distributions of two LDA topics.
    """
    if num_words is None:
        num_words = len(lda_model.id2word)

    # Get word distributions across the vocabulary for both topics
    term_dist_1 = lda_model.get_topic_terms(topic_id_1, topn=num_words)
    term_dist_2 = lda_model.get_topic_terms(topic_id_2, topn=num_words)

    # Reconstruct dense probability vectors mapping word_id -> probability
    p = np.zeros(len(lda_model.id2word))
    q = np.zeros(len(lda_model.id2word))

    for word_id, prob in term_dist_1:
        p[word_id] = prob
    for word_id, prob in term_dist_2:
        q[word_id] = prob

    # scipy returns the square root (JS Distance). Square it to get pure Divergence bounded [0, 1]
    js_distance = jensenshannon(p, q, base=2)
    js_divergence = js_distance ** 2

    return js_divergence

def evaluate_embedding_space_cca(embeddings_model1, embeddings_model2, n_components=2):
    """
    Performs CCA between two different LLM embedding configurations
    for the exact same dataset array.
    """
    # Initialize CCA configuration
    cca = CCA(n_components=n_components)

    # Fit and transform both embedding spaces into maximized canonical space
    X_c, Y_c = cca.fit_transform(embeddings_model1, embeddings_model2)

    # Track the canonical correlations across components
    correlations = []
    for i in range(n_components):
        corr_matrix = np.corrcoef(X_c[:, i], Y_c[:, i])
        # Extract the off-diagonal correlation value
        correlations.append(corr_matrix[0, 1])

    return correlations

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
                topics = lda_instance.lda_model.print_topics(-1)

                # Run classification for each model using the correct instance, texts, and corpus
                print(f"  Running DeBERTa classification for {lang}...")
                deberta_assignments = deberta.lda_category(data=texts, lda_model=lda_instance.lda_model, corpus=lda_instance.corpus)
                deberta_assignment.extend(deberta_assignments)

                print(f"  Running Qwen classification for {lang}...")
                qwen_assignments = qwen.lda_category(data=texts, lda_model=lda_instance.lda_model, corpus=lda_instance.corpus)
                qwen_assignment.extend(qwen_assignments)

                print(f"  Running XLM-RoBERTa classification for {lang}...")
                roberta_assignments = roberta.lda_category(data=texts, lda_model=lda_instance.lda_model, corpus=lda_instance.corpus)
                roberta_assignment.extend(roberta_assignments)

            except Exception as e:
                print(f"  Could not train/categorize LDA for {lang}: {e}")
        topic_counts_deberta = Counter(deberta_assignment)
        topic_counts_roberta = Counter(roberta_assignment)
        topic_counts_qwen = Counter(qwen_assignment)

        # Sort them by topic ID for plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        counts_dict = {
            "DeBERTa": topic_counts_deberta,
            "XLM-RoBERTa": topic_counts_roberta,
            "Qwen": topic_counts_qwen
        }

        for ax, (model_name, t_counts) in zip(axes, counts_dict.items()):
            topics = sorted(t_counts.keys())
            frequencies = [t_counts[t] for t in topics]

            ax.bar(topics, frequencies, color='skyblue')
            ax.set_title(f"{model_name} Topic Distribution")
            ax.set_xlabel("Topic ID")
            ax.set_ylabel("Number of Stories")
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig("lda_topic_dist.png")
        print("finished topic distrinution bar graph at: lda_topic_dist.png")

        jsd_val = calculate_topic_jsd(lda_instance.lda_model, topic_id_1=0, topic_id_2=1)
        print(f"[{lang}] JSD between Topic 0 and Topic 1: {jsd_val:.4f}")

        for lang, texts in data_dict.items():
            if not texts:
                continue

            deberta_embs = deberta.get_embeddings(texts)
            roberta_embs = roberta.get_embeddings(texts)
            qwen_embs = qwen.get_embeddings(texts)
            cca_corrs_dq = evaluate_embedding_space_cca(deberta_embs, qwen_embs, n_components=3)
            cca_corrs_dr = evaluate_embedding_space_cca(deberta_embs, roberta_embs, n_components=3)
            print(f"[{lang}] DeBERTa vs Qwen CCA Correlations: {cca_corrs_dq}")
            print(f"[{lang}] DeBERTa vs XLM-RoBERTa CCA Correlations: {cca_corrs_dr}")



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
