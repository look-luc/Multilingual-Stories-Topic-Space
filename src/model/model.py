import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class LDA:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained("NicolaiSivesind/LDA-Classifier")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "NicolaiSivesind/LDA-Classifier",
            device_map=self.device
        )
    def visualize_embeddings(
        self,
        data,
        lang:str,
        output_name:str = "lda_embedding_space"
    ):
        embeddings = []
        label_list = []
        tokenized_docs = []

        for element in data:
            tokens = self.tokenizer(
                element,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            tokens = tokens.to(self.device)

            outputs = self.model(**tokens, output_hidden_states=True)
            pooled_embedding = torch.mean(outputs.hidden_states[-1], dim=-1)

            embeddings.append(pooled_embedding.detach().cpu().numpy())
            label_list.append(lang)
            tokenized_docs.append(element.lower().split())

        embedding_matrix = np.concatenate(embeddings, axis=0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 6))
        plt.title(f'LDA embedding space for {lang}', fontsize=14)
        plt.scatter(label_list, X_reduced)
        plt.xlabel("Language")
        plt.ylabel("Space")

        plt.savefig(f"{output_name}.png", dpi=300, bbox_inches="tight")

        if torch.unique(label_list) > 1:
            silhouette_scores = silhouette_score(
                embedding_matrix,
                label_list,
                metric='euclidean'
            )
        else:
            pseudo_cluster = KMeans(
                n_clusters=3,
                random_state=42
            )
            pseudo_labels = pseudo_cluster.fit_predict(embedding_matrix)
            silhouette_scores = silhouette_score(
                embedding_matrix,
                pseudo_labels,
                metric='euclidean'
            )
        print(f"Silhouette Score: {silhouette_score:.4f}")

class XLM_RoBerta:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            "FacebookAI/xlm-roberta-base"
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            "FacebookAI/xlm-roberta-base",
            device_map="auto",
            attn_implementation="sdpa"
        )

        def visualize_embeddings(
            self,
            data,
            lang:str,
            output_name:str = "lda_embedding_space"
        ):
            embeddings = []
            label_list = []
            tokenized_docs = []

            for element in data:
                tokens = self.tokenizer(
                    element,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                tokens = tokens.to(self.device)

                outputs = self.model(**tokens, output_hidden_states=True)
                pooled_embedding = torch.mean(outputs.hidden_states[-1], dim=-1)

                embeddings.append(pooled_embedding.detach().cpu().numpy())
                label_list.append(lang)
                tokenized_docs.append(element.lower().split())

            embedding_matrix = np.concatenate(embeddings, axis=0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(embedding_matrix)
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X_scaled)

            plt.figure(figsize=(10, 6))
            plt.title(f'LDA embedding space for {lang}', fontsize=14)
            plt.scatter(label_list, X_reduced)
            plt.xlabel("Language")
            plt.ylabel("Space")

            plt.savefig(f"{output_name}.png", dpi=300, bbox_inches="tight")

            if torch.unique(label_list) > 1:
                silhouette_scores = silhouette_score(
                    embedding_matrix,
                    label_list,
                    metric='euclidean'
                )
            else:
                pseudo_cluster = KMeans(
                    n_clusters=3,
                    random_state=42
                )
                pseudo_labels = pseudo_cluster.fit_predict(embedding_matrix)
                silhouette_scores = silhouette_score(
                    embedding_matrix,
                    pseudo_labels,
                    metric='euclidean'
                )
            print(f"Silhouette Score: {silhouette_scores:.4f}")
