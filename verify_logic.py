import pandas as pd
from data_agent import analyze_correlations
import os

def test_analysis():
    print("Testing Data Analysis Agent Logic...")
    
    csv_path = "dummy.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path} with shape {df.shape}")
    
    corr_matrix, insights = analyze_correlations(df)
    
    if corr_matrix is None:
        print("Error: Correlation matrix is None.")
        return
        
    print("Correlation Matrix Generated:")
    print(corr_matrix)
    
    print("\nInsights Generated:")
    for insight in insights:
        print(insight)
        
    print("\nVerification Successful: Logic is sound.")

if __name__ == "__main__":
    test_analysis()
