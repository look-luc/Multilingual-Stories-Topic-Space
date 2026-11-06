import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModel,
    AutoTokenizer,
)


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
                embeddings.append(pooled_embedding.detach().cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def visualize_embeddings(self, data, lang: str, output_name: str = "deberta_embedding_space"):
        embedding_matrix = self.get_embeddings(data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

        pseudo_cluster = KMeans(n_clusters=min(3, len(data)), random_state=42)
        pseudo_labels = pseudo_cluster.fit_predict(embedding_matrix)
        score = silhouette_score(embedding_matrix, pseudo_labels, metric='euclidean')
        print(f"[{lang}] DeBERTa Silhouette Score: {score:.4f}")


class Qwen:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True) # Updated to a valid text embedding/LLM base model if needed
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
                embeddings.append(pooled_embedding.detach().cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def visualize_embeddings(self, data, lang: str, output_name: str = "qwen_embedding_space"):
        embedding_matrix = self.get_embeddings(data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

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
                embeddings.append(pooled_embedding.detach().cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def visualize_embeddings(self, data, lang: str, output_name: str = "xlmr_embedding_space"):
        embedding_matrix = self.get_embeddings(data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

        pseudo_cluster = KMeans(n_clusters=min(3, len(data)), random_state=42)
        pseudo_labels = pseudo_cluster.fit_predict(embedding_matrix)
        score = silhouette_score(embedding_matrix, pseudo_labels, metric='euclidean')
        print(f"[{lang}] XLM-RoBERTa Silhouette Score: {score:.4f}")


def visualize_all_models_single_plot(data, lang: str, models_dict: dict, output_name: str = "combined_models_embedding_space"):
    plt.figure(figsize=(10, 6))
    colors = {
        "DeBERTa": "tab:blue",
        "Qwen": "tab:orange",
        "XLM-RoBERTa": "tab:green"
    }

    for model_name, model_instance in models_dict.items():
        embedding_matrix = model_instance.get_embeddings(data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

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
