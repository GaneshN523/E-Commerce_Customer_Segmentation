# E-Commerce Customer Segmentation Engine

This interactive web application demonstrates how to perform customer segmentation using unsupervised learning. Built with Python and Streamlit, it covers every step from data loading and pre‑processing to clustering (using K‑Means and DBSCAN) and dimensionality reduction (using PCA and t-SNE). Detailed explanations and interactive elements are provided to help users and developers understand the complete flow of processing customer data.

## Features

- **Data Loading & Pre-Processing:**  
  Load a CSV file containing customer data, clean the data, select numerical features, and apply normalization. Learn how to convert text data into numerical representation if required.

- **Clustering:**  
  Explore customer segmentation with two popular clustering algorithms:  
  - **K-Means:** Understand the iterative process of centroid initialization, assignment, and update.  
  - **DBSCAN:** Learn how density-based clustering works to discover clusters and identify noise.

- **Dimensionality Reduction & Visualization:**  
  Visualize high-dimensional data in 2D using:  
  - **PCA:** Project data onto principal components and visualize the spread.  
  - **t-SNE:** Explore nonlinear relationships by reducing dimensions with t-SNE.

- **Interactive Explanations:**  
  Each step includes detailed markdown explanations and expandable sections so that users can follow the underlying logic and learn how each processing step works.

## Directory Structure

```plaintext
customer_segmentation/
├── app.py                   # Main Streamlit application file.
├── README.md                # Project documentation.
├── requirements.txt         # List of required Python libraries.
├── data/
│   └── customers.csv        # Sample CSV data file.
└── utils/
    ├── __init__.py
    ├── data_loader.py       # Data loading and pre-processing functions.
    ├── clustering.py        # Clustering functions (K-Means & DBSCAN).
    └── visualization.py     # Dimensionality reduction and plotting functions.
