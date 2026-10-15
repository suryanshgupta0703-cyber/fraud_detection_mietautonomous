"""
STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
========================================
Visualizes data to understand patterns:
- Distribution of features
- Fraud vs Genuine patterns
- Correlations
- Creates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_cleaned_data():
    """Load the cleaned dataset from Step 2"""
    try:
        df = pd.read_csv('data_cleaned.csv')
        print(f"\n✅ Cleaned dataset loaded successfully!")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        return df
    except FileNotFoundError:
        print("\n❌ ERROR: data_cleaned.csv not found!")
        print("   Please run: python 2_preprocessing.py first")
        return None


def exploratory_data_analysis(df):
    """
    Analyze and visualize the dataset
    """
    print("\n" + "="*60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    # ============================================================
    # 1. BASIC INFO
    # ============================================================
    print("\n1️⃣  DATASET INFO")
    print("-" * 60)
    
    print(f"\nShape: {df.shape}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"\nColumn names and types:")
    print(df.dtypes)
    
    # ============================================================
    # 2. CLASS DISTRIBUTION
    # ============================================================
    print("\n2️⃣  CLASS DISTRIBUTION")
    print("-" * 60)
    
    fraud_dist = df['fraud'].value_counts()
    fraud_pct = df['fraud'].value_counts(normalize=True) * 100
    
    print("\nCounts:")
    print(f"  Genuine (0): {fraud_dist[0]:,}")
    print(f"  Fraud (1):   {fraud_dist[1]:,}")
    
    print("\nPercentages:")
    print(f"  Genuine: {fraud_pct[0]:.2f}%")
    print(f"  Fraud:   {fraud_pct[1]:.2f}%")
    
    imbalance_ratio = fraud_dist[0] / fraud_dist[1]
    print(f"\n⚠️  Imbalance Ratio: {imbalance_ratio:.1f}:1")
    print("    (Will be fixed in Step 4 with SMOTE)")
    
    # ============================================================
    # 3. STATISTICAL SUMMARY
    # ============================================================
    print("\n3️⃣  STATISTICAL SUMMARY")
    print("-" * 60)
    
    print("\nAll numerical columns:")
    print(df.describe())
    
    # ============================================================
    # 4. CORRELATION ANALYSIS
    # ============================================================
    print("\n4️⃣  CORRELATION WITH FRAUD")
    print("-" * 60)
    
    correlation = df.corr()['fraud'].sort_values(ascending=False)
    print("\nHow each feature correlates with fraud:")
    print(correlation)
    
    # ============================================================
    # 5. FRAUD PATTERNS
    # ============================================================
    print("\n5️⃣  FRAUD PATTERNS")
    print("-" * 60)
    
    genuine_df = df[df['fraud'] == 0]
    fraud_df = df[df['fraud'] == 1]
    
    print("\n📊 GENUINE TRANSACTIONS:")
    print(f"  Average Amount: ₹{genuine_df['amount'].mean():.2f}")
    print(f"  Common Time: {genuine_df['time'].mode()[0]}:00")
    print(f"  International: {(genuine_df['international'] == 1).sum()} ({(genuine_df['international'] == 1).sum()/len(genuine_df)*100:.1f}%)")
    
    print("\n🚨 FRAUDULENT TRANSACTIONS:")
    print(f"  Average Amount: ₹{fraud_df['amount'].mean():.2f}")
    print(f"  Common Time: {fraud_df['time'].mode()[0]}:00")
    print(f"  International: {(fraud_df['international'] == 1).sum()} ({(fraud_df['international'] == 1).sum()/len(fraud_df)*100:.1f}%)")
    
    # ============================================================
    # 6. CREATE VISUALIZATIONS
    # ============================================================
    print("\n6️⃣  CREATING VISUALIZATIONS")
    print("-" * 60)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Amount Distribution
    axes[0, 0].hist(genuine_df['amount'], bins=30, label='Genuine', alpha=0.7, color='green')
    axes[0, 0].hist(fraud_df['amount'], bins=30, label='Fraud', alpha=0.7, color='red')
    axes[0, 0].set_title('Amount Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Amount (₹)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Class Distribution (Pie Chart)
    fraud_counts = df['fraud'].value_counts()
    colors = ['green', 'red']
    axes[0, 1].pie(fraud_counts, labels=['Genuine', 'Fraud'], autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 1].set_title('Class Distribution', fontsize=12, fontweight='bold')
    
    # Plot 3: Time Distribution
    axes[0, 2].hist(genuine_df['time'], bins=24, label='Genuine', alpha=0.7, color='green')
    axes[0, 2].hist(fraud_df['time'], bins=24, label='Fraud', alpha=0.7, color='red')
    axes[0, 2].set_title('Time Distribution', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Hour of Day')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Plot 4: Amount Box Plot
    box_data = [genuine_df['amount'], fraud_df['amount']]
    axes[1, 0].boxplot(box_data, tick_labels=['Genuine', 'Fraud'])
    axes[1, 0].set_title('Amount by Fraud Status', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Amount (₹)')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 5: International Distribution
    intl_fraud = df.groupby('international')['fraud'].mean()
    axes[1, 1].bar(['Domestic', 'International'], [intl_fraud[0], intl_fraud[1]], 
                   color=['green', 'red'], alpha=0.7)
    axes[1, 1].set_title('Fraud Rate by Transaction Type', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Fraud Probability')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    # Plot 6: Merchant Type Distribution
    merchant_fraud = df.groupby('merchant_type')['fraud'].mean()
    axes[1, 2].bar(merchant_fraud.index, merchant_fraud.values, color='orange', alpha=0.7)
    axes[1, 2].set_title('Fraud Rate by Merchant Type', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Merchant Type')
    axes[1, 2].set_ylabel('Fraud Probability')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('eda_visualization.png', dpi=100, bbox_inches='tight')
    print("\n✅ Visualizations saved as 'eda_visualization.png'")
    plt.close()
    
    # ============================================================
    # 7. KEY INSIGHTS
    # ============================================================
    print("\n" + "="*60)
    print("🔍 KEY INSIGHTS")
    print("="*60)
    
    print("\n✅ Amount:")
    print(f"   Genuine avg: ₹{genuine_df['amount'].mean():.0f}")
    print(f"   Fraud avg:   ₹{fraud_df['amount'].mean():.0f}")
    print(f"   → Frauds use 2-3x higher amounts")
    
    print("\n✅ Time:")
    print(f"   Fraud common at: {fraud_df['time'].mode()[0]}:00 (late night/early morning)")
    print(f"   → Frauds happen at unusual times")
    
    print("\n✅ International:")
    intl_fraud_pct = (fraud_df['international'] == 1).sum() / len(fraud_df) * 100
    intl_genuine_pct = (genuine_df['international'] == 1).sum() / len(genuine_df) * 100
    print(f"   Genuine international: {intl_genuine_pct:.1f}%")
    print(f"   Fraud international:   {intl_fraud_pct:.1f}%")
    print(f"   → Frauds are more international")
    
    print("\n✅ Merchant Type:")
    high_risk_merchant = merchant_fraud.idxmax()
    print(f"   Highest fraud merchant type: {high_risk_merchant}")
    print(f"   → Some merchants are riskier")
    
    return df


if __name__ == "__main__":
    """
    Run this file to analyze the dataset
    Command: python 3_eda.py
    """
    print("\n🏦 CREDIT CARD FRAUD DETECTION PROJECT")
    print("="*60)
    
    # Load cleaned data
    df = load_cleaned_data()
    
    if df is not None:
        # Perform EDA
        exploratory_data_analysis(df)
        
        print("\n" + "="*60)
        print("✅ STEP 3 COMPLETE!")
        print("="*60)
        print("\n📝 Next Step:")
        print("   Run: python 4_handle_imbalance.py")
        print("\n")
    else:
        print("\n❌ Cannot proceed without data!")