import streamlit as st
import pandas as pd
from utils.data_loader import load_data, preprocess_data, explain_text_to_numeric
from utils.clustering import perform_kmeans, perform_dbscan
from utils.visualization import plot_pca, plot_tsne
import matplotlib.pyplot as plt

# Set up the page configuration
st.set_page_config(page_title="E-Commerce Customer Segmentation Engine", layout="wide")

st.title("E-Commerce Customer Segmentation Engine")
st.markdown("""
This interactive application guides you through every step of a customer segmentation project.
We will cover:
- **Data Loading & Pre-Processing:** How we load and clean the data.
- **Text Data Conversion:** How text data (if any) can be converted to numerical form.
- **Clustering:** How to segment customers using **K-Means** and **DBSCAN**.
- **Dimensionality Reduction:** How to visualize high-dimensional data using **PCA** and **t-SNE**.

Each section includes detailed explanations of the logic and methods used.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:",
                        ("Data Loading & Pre-Processing", "Clustering", "Dimensionality Reduction"))

# Path to the CSV data file
data_path = "data/customers.csv"

# Load the data (raw) and preprocessed version (explained in the function)
df_raw = load_data(data_path)
df_processed = preprocess_data(df_raw)

# Detailed explanation for text data conversion (if applicable)
with st.expander("Learn About Converting Text Data to Numeric"):
    st.markdown("""
    **Text to Numeric Conversion:**  
    In many customer datasets, you may encounter categorical (text) features, such as:
    - **Gender:** e.g. 'Male', 'Female'
    - **City/Region:** e.g. 'New York', 'Los Angeles'
    
    **How to convert?**  
    - **Label Encoding:** Assigns a unique number to each category. Useful when categories have an inherent order.
    - **One-Hot Encoding:** Creates binary columns for each category. Preferred when there is no natural ordering.
    
    In this project, our sample data is numerical (age, annual_income, spending_score), but if you had text data you could do something like:
    
    ```python
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # Example for label encoding:
    le = LabelEncoder()
    df['gender_numeric'] = le.fit_transform(df['gender'])
    
    # Example for one-hot encoding:
    df = pd.get_dummies(df, columns=['city'])
    ```
    
    The helper function `explain_text_to_numeric` (see `utils/data_loader.py`) outlines this process.
    """)

if page == "Data Loading & Pre-Processing":
    st.header("Step 1: Data Loading & Pre-Processing")
    
    # Explain Data Loading
    st.markdown("### Loading the Data")
    st.write("The raw data is loaded from the CSV file. Below is a preview of the raw data:")
    st.dataframe(df_raw.head())

    # Explanation of Data Preprocessing
    st.markdown("### Pre-Processing the Data")
    st.write("In this step, we perform several tasks:")
    st.markdown("""
    - **Cleaning:** Removing rows with missing values.
    - **Feature Selection:** Keeping only the necessary numerical features.
    - **Normalization:** Standardizing the features using StandardScaler.
    """)
    st.write("Preview of pre-processed data:")
    st.dataframe(df_processed.head())

    with st.expander("Detailed Explanation of Data Pre-Processing Code"):
        st.markdown("""
        1. **Data Cleaning:**  
           We drop any rows that contain missing values to ensure our analysis isn’t affected by incomplete data.
           
        2. **Feature Selection:**  
           We assume our dataset has columns: `age`, `annual_income`, and `spending_score`. These are selected for clustering.
           
        3. **Normalization:**  
           We use `StandardScaler` from scikit-learn to normalize the data. This ensures that each feature contributes equally to the analysis.
           
        4. **Text Data Conversion:**  
           Although our current dataset is numeric, if you had categorical columns, you could convert them using techniques like Label Encoding or One-Hot Encoding.
        """)
        
elif page == "Clustering":
    st.header("Step 2: Customer Clustering")
    st.markdown("This section demonstrates how to segment customers using clustering algorithms.")
    
    st.markdown("### Choose a Clustering Algorithm")
    algo = st.selectbox("Select Clustering Algorithm", ("K-Means", "DBSCAN"))
    
    if algo == "K-Means":
        k = st.slider("Number of Clusters (k)", 2, 10, 3)
        df_clustered = perform_kmeans(df_processed, n_clusters=k)
        st.write(f"#### Data with K-Means Clusters (k = {k})")
        st.dataframe(df_clustered.head())
        
        with st.expander("How K-Means Works"):
            st.markdown("""
            **K-Means Clustering:**  
            - **Initialization:** Randomly choose `k` centroids.
            - **Assignment:** Each data point is assigned to the nearest centroid.
            - **Update:** The centroids are recomputed as the mean of the points assigned to them.
            - **Repeat:** The assignment and update steps are repeated until convergence.
            """)
        
    else:
        eps = st.slider("DBSCAN eps parameter", 0.1, 10.0, 1.0)
        min_samples = st.slider("DBSCAN min_samples", 3, 10, 5)
        df_clustered = perform_dbscan(df_processed, eps=eps, min_samples=min_samples)
        st.write(f"#### Data with DBSCAN Clusters (eps = {eps}, min_samples = {min_samples})")
        st.dataframe(df_clustered.head())
        
        with st.expander("How DBSCAN Works"):
            st.markdown("""
            **DBSCAN Clustering:**  
            - **Core Points:** A point is considered a core point if it has at least `min_samples` neighbors within a distance `eps`.
            - **Cluster Formation:** Core points and their neighbors form a cluster.
            - **Noise:** Points not belonging to any cluster are labeled as noise.
            """)
    
elif page == "Dimensionality Reduction":
    st.header("Step 3: Dimensionality Reduction & Visualization")
    st.markdown("We reduce the high-dimensional data to 2 dimensions to visualize clusters and patterns.")
    
    st.markdown("### Choose a Reduction Technique")
    reduction_algo = st.selectbox("Select Dimensionality Reduction Technique", ("PCA", "t-SNE"))
    
    if reduction_algo == "PCA":
        fig = plot_pca(df_processed)
        st.pyplot(fig)
        
        with st.expander("How PCA Works"):
            st.markdown("""
            **Principal Component Analysis (PCA):**  
            - **Goal:** Find the directions (principal components) that maximize the variance in the data.
            - **Process:**  
              1. Compute the covariance matrix.
              2. Calculate the eigenvectors and eigenvalues.
              3. Project the data onto the top components.
            - **Visualization:** The first two components are plotted to visualize the data in 2D.
            """)
            
    else:
        perplexity = st.slider("t-SNE Perplexity", 5, 50, 30)
        fig = plot_tsne(df_processed, perplexity=perplexity)
        st.pyplot(fig)
        
        with st.expander("How t-SNE Works"):
            st.markdown("""
            **t-SNE (t-Distributed Stochastic Neighbor Embedding):**  
            - **Goal:** Visualize high-dimensional data by modeling each high-dimensional object by a two- or three-dimensional point.
            - **Process:**  
              1. Compute pairwise similarities in high dimensions.
              2. Compute pairwise similarities in low dimensions.
              3. Minimize the difference between these similarities.
            - **Parameter:** The `perplexity` parameter relates to the number of effective nearest neighbors.
            """)

st.sidebar.markdown("---")
st.sidebar.info("This tool is built to help marketing teams and developers understand every detail of the customer segmentation process.")
