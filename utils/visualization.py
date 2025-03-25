import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

def plot_pca(df: pd.DataFrame):
    """
    Reduce the data to 2 principal components using PCA and create a scatter plot.
    
    Steps:
    1. Initialize PCA with 2 components.
    2. Transform the data to get the principal components.
    3. Plot the two components on a 2D scatter plot.
    """
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(df)
    
    fig, ax = plt.subplots()
    ax.scatter(components[:, 0], components[:, 1], c='blue', alpha=0.5)
    ax.set_title("PCA: 2 Component Projection")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    return fig

def plot_tsne(df: pd.DataFrame, perplexity: int = 30):
    """
    Reduce the data to 2 dimensions using t-SNE and create a scatter plot.
    
    Steps:
    1. Initialize t-SNE with 2 components and the given perplexity.
    2. Transform the data to get the 2D embedding.
    3. Plot the embedded data on a scatter plot.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    components = tsne.fit_transform(df)
    
    fig, ax = plt.subplots()
    ax.scatter(components[:, 0], components[:, 1], c='green', alpha=0.5)
    ax.set_title("t-SNE: 2 Component Projection")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    return fig
