from src.embedding.positional_encoder import PositionalEncoding
from src.embedding.embedding_table import EmbeddingTable
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_position_encoder(pe: PositionalEncoding):
    plt.figure(figsize=(10, 8))
    sns.heatmap(pe.pe.squeeze(0).cpu().numpy(), cmap="coolwarm", cbar=True)
    plt.title('Positional Encoding Heatmap')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Sequence Position')
    plt.show()

def visualize_embedding_table(et: EmbeddingTable):
    embedding_weights = et.et.cpu().numpy()  # Shape: (vocab_size, d_model)

    # Plot as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(embedding_weights, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

    # Labels
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Token Index")
    plt.title("Embedding Table Heatmap")
    plt.show()