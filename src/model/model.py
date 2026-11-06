import matplotlib.pyplot as plt
import numpy as np
import torch
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModel,
    AutoTokenizer,
)


def visualize_all_models_single_plot(
    data,
    lang: str,
    models_dict: dict,
    output_name: str = "combined_models_embedding_space"
):
    """
    Runs text data through multiple models and plots all their PCA-reduced
    embeddings together on a single scatterplot.
    """
    plt.figure(figsize=(10, 6))

    # Define distinct colors for each model
    colors = {
        "DeBERTa": "tab:blue",
        "Qwen": "tab:orange",
        "XLM-RoBERTa": "tab:green"
    }

    for model_name, model_instance in models_dict.items():
        embeddings = []

        for element in data:
            tokens = model_instance.tokenizer(
                element,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            tokens = {k: v.to(model_instance.device) for k, v in tokens.items()}

            outputs = model_instance.model(**tokens, output_hidden_states=True)
            pooled_embedding = torch.mean(outputs.hidden_states[-1], dim=1)

            embeddings.append(pooled_embedding.detach().cpu().numpy())

        embedding_matrix = np.concatenate(embeddings, axis=0)

        # Scale and reduce dimensions for this model's space
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

        # Plot all points on the same graph
        plt.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            alpha=0.6,
            label=model_name,
            color=colors.get(model_name, "tab:purple")
        )

    plt.title(f'Combined Embedding Space Comparison ({lang})', fontsize=14)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Models")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(f"./{output_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined scatterplot for {lang} to {output_name}.png")

def compute_topic_coherence(texts, top_words_per_topic):
    """
    Computes C_v topic coherence given tokenized texts and extracted topic words.

    :param texts: List of raw text strings
    :param top_words_per_topic: List of lists containing top words for each topic e.g., [['story', 'king', 'gold'], ...]
    """
    tokenized_docs = [simple_preprocess(text) for text in texts]

    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    coherence_model = CoherenceModel(
        topics=top_words_per_topic,
        texts=tokenized_docs,
        dictionary=dictionary,
        corpus=corpus,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()
    print(f"Topic Coherence (C_v): {coherence_score:.4f}")
    return coherence_score

def evaluate_cross_lingual_similarity(embeddings_lang1, embeddings_lang2):
    """
    Computes the cosine similarity matrix between two language embedding spaces.
    """
    similarity_matrix = cosine_similarity(embeddings_lang1, embeddings_lang2)

    mean_similarity = np.mean(similarity_matrix)
    print(f"Mean Cross-Lingual Cosine Similarity: {mean_similarity:.4f}")
    return similarity_matrix

class DeBerta:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
        self.model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
        self.model.to(self.device)

    def visualize_embeddings(
        self,
        data,
        lang: str,
        output_name: str = "deberta_embedding_space"
    ):
        embeddings = []
        label_list = []

        for element in data:
            tokens = self.tokenizer(
                element,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            outputs = self.model(**tokens, output_hidden_states=True)
            pooled_embedding = torch.mean(outputs.hidden_states[-1], dim=1)

            embeddings.append(pooled_embedding.detach().cpu().numpy())
            label_list.append(lang)

        embedding_matrix = np.concatenate(embeddings, axis=0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

        if len(set(label_list)) > 1:
            score = silhouette_score(embedding_matrix, label_list, metric='euclidean')
        else:
            pseudo_cluster = KMeans(n_clusters=3, random_state=42)
            pseudo_labels = pseudo_cluster.fit_predict(embedding_matrix)
            score = silhouette_score(embedding_matrix, pseudo_labels, metric='euclidean')

        print(f"[{lang}] DeBERTa Silhouette Score: {score:.4f}")

class Qwen:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-Embedding-8B", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-8B", trust_remote_code=True)
        self.model.to(self.device)

    def visualize_embeddings(
        self,
        data,
        lang: str,
        output_name: str = "qwen_embedding_space"
    ):
        embeddings = []
        label_list = []

        for element in data:
            tokens = self.tokenizer(
                element,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            outputs = self.model(**tokens, output_hidden_states=True)
            pooled_embedding = torch.mean(outputs.hidden_states[-1], dim=1)

            embeddings.append(pooled_embedding.detach().cpu().numpy())
            label_list.append(lang)

        embedding_matrix = np.concatenate(embeddings, axis=0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

        if len(set(label_list)) > 1:
            score = silhouette_score(embedding_matrix, label_list, metric='euclidean')
        else:
            pseudo_cluster = KMeans(n_clusters=3, random_state=42)
            pseudo_labels = pseudo_cluster.fit_predict(embedding_matrix)
            score = silhouette_score(embedding_matrix, pseudo_labels, metric='euclidean')

        print(f"[{lang}] Qwen Silhouette Score: {score:.4f}")


class XLM_RoBerta:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        self.model = AutoModel.from_pretrained(
            "FacebookAI/xlm-roberta-base",
            attn_implementation="sdpa"
        )
        self.model.to(self.device)

    def visualize_embeddings(
        self,
        data,
        lang: str,
        output_name: str = "xlmr_embedding_space"
    ):
        embeddings = []
        label_list = []

        for element in data:
            tokens = self.tokenizer(
                element,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            outputs = self.model(**tokens, output_hidden_states=True)
            pooled_embedding = torch.mean(outputs.hidden_states[-1], dim=1)

            embeddings.append(pooled_embedding.detach().cpu().numpy())
            label_list.append(lang)

        embedding_matrix = np.concatenate(embeddings, axis=0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

        if len(set(label_list)) > 1:
            score = silhouette_score(embedding_matrix, label_list, metric='euclidean')
        else:
            pseudo_cluster = KMeans(n_clusters=3, random_state=42)
            pseudo_labels = pseudo_cluster.fit_predict(embedding_matrix)
            score = silhouette_score(embedding_matrix, pseudo_labels, metric='euclidean')

        print(f"[{lang}] XLM-RoBERTa Silhouette Score: {score:.4f}")
