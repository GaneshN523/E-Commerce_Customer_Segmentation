import pandas as pd
from sklearn.cluster import KMeans, DBSCAN

def perform_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Apply K-Means clustering on the data and add a new column 'cluster'
    representing the cluster assignment for each row.
    
    Detailed steps:
    1. Initialize the KMeans object with the desired number of clusters.
    2. Fit the model to the data and predict cluster labels.
    3. Append the labels to the original data.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df)
    df_clustered = df.copy()
    df_clustered["cluster"] = clusters
    return df_clustered

def perform_dbscan(df: pd.DataFrame, eps: float = 1.0, min_samples: int = 5) -> pd.DataFrame:
    """
    Apply DBSCAN clustering on the data and add a new column 'cluster'
    representing the cluster assignment for each row.
    
    Detailed steps:
    1. Initialize the DBSCAN object with `eps` and `min_samples` parameters.
    2. Fit the model to the data and predict cluster labels.
    3. Append the labels to the original data.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df)
    df_clustered = df.copy()
    df_clustered["cluster"] = clusters
    return df_clustered
