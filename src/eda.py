import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(filepath, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(filepath)
    
    # Basic Info
    print("Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Target Variable Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Avg Salary(K)'], kde=True)
    plt.title('Distribution of Average Salary')
    plt.xlabel('Average Salary (K)')
    plt.savefig(f'{output_dir}/salary_distribution.png')
    plt.close()
    
    # Correlation Matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(f'{output_dir}/correlation_matrix.png')
    plt.close()

    # Job Title Count
    plt.figure(figsize=(10, 6))
    df['job_title_sim'].value_counts().plot(kind='bar')
    plt.title('Job Title Distribution')
    plt.xlabel('Job Title')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/job_title_distribution.png')
    plt.close()

if __name__ == "__main__":
    run_eda('project_dataset.csv')
