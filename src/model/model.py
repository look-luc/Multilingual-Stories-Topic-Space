import matplotlib.pyplot as plt
import numpy as np
import torch
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
    Dynamically maps all models to unique colors and all languages to unique marker shapes
    using a unified global projection space.
    """
    plt.figure(figsize=(12, 8))

    model_names = list(models_dict.keys())
    lang_names = list(langs_data.keys())

    model_colors = plt.get_cmap('tab10', max(len(model_names), 1))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P']

    # 1. Collect all model embeddings globally to fit a single shared PCA/StandardScaler space
    global_embeddings = []
    model_data_tracking = {}

    for model_name, model_instance in models_dict.items():
        model_embeddings = {}
        for lang_name, texts in langs_data.items():
            if not texts:
                continue
            emb = model_instance.get_embeddings(texts)
            model_embeddings[lang_name] = emb
            global_embeddings.append(emb)
        model_data_tracking[model_name] = model_embeddings

    # Fit global scaler and PCA so all models share identical 2D projection bounds
    global_matrix = np.concatenate(global_embeddings, axis=0)
    scaler = StandardScaler()
    scaler.fit(global_matrix)
    pca = PCA(n_components=2)
    pca.fit(scaler.transform(global_matrix))

    # 2. Plot each model/language configuration onto the shared space
    for model_idx, (model_name, model_embeddings) in enumerate(model_data_tracking.items()):
        model_color = model_colors(model_idx)

        for lang_idx, lang_name in enumerate(lang_names):
            if lang_name not in model_embeddings:
                continue

            emb = model_embeddings[lang_name]
            X_reduced = pca.transform(scaler.transform(emb))
            marker = markers[lang_idx % len(markers)]

            plt.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                alpha=0.6,
                color=model_color,
                marker=marker,
                label=f"{model_name} ({lang_name})"
            )

    plt.title("Cross-Lingual & Cross-Model Global Embedding Space Comparison", fontsize=14)
    plt.xlabel("Global PCA Component 1")
    plt.ylabel("Global PCA Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper center', fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"./{output_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined multilingual scatterplot to {output_name}.png")
