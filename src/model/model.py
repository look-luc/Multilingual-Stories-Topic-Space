import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler  # type: ignore
from transformers import (
    AutoModel,
    AutoTokenizer,
)


def evaluate_cross_lingual_similarity(embeddings_lang1, embeddings_lang2):
    """
    Computes the cosine similarity matrix between two language embedding spaces.
    """
    similarity_matrix = cosine_similarity(embeddings_lang1, embeddings_lang2)
    return similarity_matrix


class DeBerta:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
        self.model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
        self.model.to(self.device)

    def get_embeddings(self, data):
        embeddings = []
        self.model.eval()
        with torch.no_grad():
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
                embeddings.append(pooled_embedding.detach().cpu().float().numpy())
        return np.concatenate(embeddings, axis=0)

    def visualize_embeddings(self, data, lang: str, output_name: str = "deberta_embedding_space"):
        embedding_matrix = self.get_embeddings(data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        pca.fit_transform(X_scaled)
        pseudo_cluster = KMeans(n_clusters=min(3, len(data)), random_state=42)
        pseudo_labels = pseudo_cluster.fit_predict(embedding_matrix)
        score = silhouette_score(embedding_matrix, pseudo_labels, metric='euclidean')
        print(f"[{lang}] DeBERTa Silhouette Score: {score:.4f}")


class Qwen:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        self.model.to(self.device)

    def get_embeddings(self, data):
        embeddings = []
        self.model.eval()
        with torch.no_grad():
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
                embeddings.append(pooled_embedding.detach().cpu().float().numpy())
        return np.concatenate(embeddings, axis=0)

    def visualize_embeddings(self, data, lang: str, output_name: str = "qwen_embedding_space"):
        embedding_matrix = self.get_embeddings(data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        pca.fit_transform(X_scaled)

        pseudo_cluster = KMeans(n_clusters=min(3, len(data)), random_state=42)
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

    def get_embeddings(self, data):
        embeddings = []
        self.model.eval()
        with torch.no_grad():
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
                embeddings.append(pooled_embedding.detach().cpu().float().numpy())
        return np.concatenate(embeddings, axis=0)

    def visualize_embeddings(self, data, lang: str, output_name: str = "xlmr_embedding_space"):
        embedding_matrix = self.get_embeddings(data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        pca.fit_transform(X_scaled)

        pseudo_cluster = KMeans(n_clusters=min(3, len(data)), random_state=42)
        pseudo_labels = pseudo_cluster.fit_predict(embedding_matrix)
        score = silhouette_score(embedding_matrix, pseudo_labels, metric='euclidean')
        print(f"[{lang}] XLM-RoBERTa Silhouette Score: {score:.4f}")


def visualize_multilingual_models_single_plot(
    langs_data: dict,
    models_dict: dict,
    output_name: str = "multilingual_models_embedding_space"
):
    """
    Visualizes multilingual embeddings for each model in separate subplots
    using UMAP for dimensionality reduction and externalizing the legend.
    """
    fig, axes = plt.subplots(1, len(models_dict), figsize=(7 * len(models_dict), 6), squeeze=False)
    axes = axes.flatten()

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P']

    for model_idx, (model_name, model_instance) in enumerate(models_dict.items()):
        ax = axes[model_idx]

        model_embeddings = []
        lang_tracking = []
        for lang_name, texts in langs_data.items():
            if not texts:
                continue
            emb = model_instance.get_embeddings(texts)
            model_embeddings.append(emb)
            lang_tracking.append(lang_name)

        if not model_embeddings:
            continue

        model_matrix = np.concatenate(model_embeddings, axis=0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(model_matrix)

        # Replaced PCA with UMAP for non-linear embedding separation
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)

        start_idx = 0
        for lang_idx, lang_name in enumerate(lang_tracking):
            texts = langs_data[lang_name]
            end_idx = start_idx + len(texts)
            lang_coords = X_reduced[start_idx:end_idx]
            start_idx = end_idx

            marker = markers[lang_idx % len(markers)]
            ax.scatter(
                lang_coords[:, 0],
                lang_coords[:, 1],
                alpha=0.6,
                s=15,
                marker=marker,
                label=lang_name
            )

        ax.set_title(f"{model_name}", fontsize=12)
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.grid(True, linestyle="--", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()

    # Create a single figure-level legend placed at the top or bottom
    fig.legend(
        handles,
        labels,
        loc='upper center',        # Position of the legend
        ncol=len(labels),          # Number of columns inside the legend box
        bbox_to_anchor=(0.5, 1.02) # Fine-tune placement above the subplots
    )
    plt.suptitle("Cross-Lingual Embedding Space Comparison Across Models (UMAP)", fontsize=14)

    # Adjust layout to provide padding for the external legends
    plt.subplots_adjust(right=0.82, wspace=0.3)
    plt.savefig(f"./{output_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined multilingual UMAP scatterplot to {output_name}.png")
