import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(filepath):
    df = pd.read_csv(filepath)
    
    # Replace -1 with NaN for proper handling, though some columns -1 might be a valid category (like 'unknown')
    # For now, let's treat -1 in numerical columns as NaN if it makes sense, or keep it if it's a flag.
    # Looking at the data, -1 in 'Founded' means unknown. 
    # -1 in 'Competitors' (if it was numeric) would be unknown.
    
    # Let's drop columns that are not useful or leaky
    # 'index' is just an ID.
    # 'Salary Estimate', 'Lower Salary', 'Upper Salary' are directly related to target.
    # 'Company Name' might be too high cardinality, but 'company_txt' seems to be the cleaned version.
    # 'Job Description' is text.
    
    drop_cols = ['index', 'Salary Estimate', 'Lower Salary', 'Upper Salary', 'Job Description']
    
    # We will process Job Description separately to extract TF-IDF features
    text_data = df['Job Description']
    
    y = df['Avg Salary(K)']
    X = df.drop(columns=drop_cols + ['Avg Salary(K)'])
    
    return X, y, text_data

def get_preprocessor(X):
    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Pipeline for numeric features
    # Impute missing values (if any) with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features
    # Impute missing with 'missing', then OneHotEncode
    # Handle unknown categories by ignoring them
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def process_text(text_data, max_features=100):
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    text_features = tfidf.fit_transform(text_data).toarray()
    feature_names = tfidf.get_feature_names_out()
    return text_features, feature_names

def prepare_data(filepath):
    X, y, text_data = load_data(filepath)
    
    # Split first to avoid data leakage
    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        X, y, text_data, test_size=0.2, random_state=42
    )
    
    # Process structured data
    preprocessor = get_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names for structured data
    # This is a bit tricky with ColumnTransformer but useful for importance
    try:
        num_names = preprocessor.named_transformers_['num'].get_feature_names_out()
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        structured_feature_names = np.concatenate([num_names, cat_names])
    except:
        structured_feature_names = [f"feat_{i}" for i in range(X_train_processed.shape[1])]

    # Process text data
    # Fit TF-IDF on train text only
    tfidf = TfidfVectorizer(stop_words='english', max_features=50)
    text_train_features = tfidf.fit_transform(text_train).toarray()
    text_test_features = tfidf.transform(text_test).toarray()
    text_feature_names = tfidf.get_feature_names_out()
    
    # Combine features
    X_train_final = np.hstack([X_train_processed, text_train_features])
    X_test_final = np.hstack([X_test_processed, text_test_features])
    
    all_feature_names = np.concatenate([structured_feature_names, text_feature_names])
    
    return X_train_final, X_test_final, y_train, y_test, all_feature_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feats = prepare_data('project_dataset.csv')
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")
