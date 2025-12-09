import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from preprocessing import prepare_data

def run_unsupervised_analysis(filepath, output_dir='plots'):
    print("Loading data for unsupervised analysis...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(filepath)
    
    # Combine train and test for unsupervised analysis to see global structure
    X_all = np.vstack([X_train, X_test])
    
    # 1. PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, c='blue', edgecolor='k')
    plt.title('PCA of Job Postings (2 Components)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.savefig(f'{output_dir}/pca_plot.png')
    print(f"PCA plot saved to {output_dir}/pca_plot.png")
    
    # 2. K-Means Clustering
    print("Running K-Means Clustering...")
    # Use Elbow method to find optimal k (simplified: just try k=3 for now)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_all)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, edgecolor='k')
    plt.title('K-Means Clustering on PCA-reduced Data')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.savefig(f'{output_dir}/kmeans_clusters.png')
    print(f"K-Means plot saved to {output_dir}/kmeans_clusters.png")
    
    # Analyze Clusters against Salary (using y_train + y_test)
    y_all = pd.concat([y_train, y_test])
    cluster_df = pd.DataFrame({'Cluster': clusters, 'Salary': y_all.values})
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Cluster', y='Salary', data=cluster_df)
    plt.title('Salary Distribution by Cluster')
    plt.savefig(f'{output_dir}/cluster_salary_boxplot.png')
    print(f"Cluster salary boxplot saved to {output_dir}/cluster_salary_boxplot.png")

if __name__ == "__main__":
    run_unsupervised_analysis('project_dataset.csv')
