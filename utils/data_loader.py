import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load customer data from CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process the data by:
    - Removing rows with missing values.
    - Selecting relevant numerical features.
    - Normalizing features using StandardScaler.
    
    If the dataset contains text data, you could first convert them to numerical values.
    """
    # Remove rows with missing values
    df_clean = df.dropna()
    
    # For this project, we assume the following columns exist:
    selected_features = ["age", "annual_income", "spending_score"]
    df_features = df_clean[selected_features].copy()
    
    # Normalize the selected features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_features[selected_features] = scaler.fit_transform(df_features[selected_features])
    
    return df_features

def explain_text_to_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    An example function to convert text data into numeric data.
    
    This function demonstrates how you can convert a categorical text column
    into numerical data using Label Encoding. In a real scenario, you may
    choose LabelEncoder, OneHotEncoder, or other methods depending on your needs.
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[column + "_numeric"] = le.fit_transform(df[column])
    return df
