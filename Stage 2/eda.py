import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for clean visual aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["figure.dpi"] = 150

class ExploratoryDataAnalysis:
    """
    A comprehensive toolkit for Exploratory Data Analysis (EDA) in Python.
    Provides automated inspection, missing data assessment, numerical/categorical summaries,
    outlier detection, correlation analysis, visualization, and HTML report export.
    """
    def __init__(self, data: pd.DataFrame | str, name: str = "Dataset"):
        if isinstance(data, str):
            self.file_path = data
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.file_path = "In-memory DataFrame"
            self.df = data.copy()
        else:
            raise ValueError("Data must be a file path (str) or pandas DataFrame.")
        
        self.name = name
        self.num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = self.df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()

    def overview(self) -> pd.DataFrame:
        """Prints & returns overall dataset metadata."""
        print("==================================================")
        print(f" EDA OVERVIEW: {self.name}")
        print("==================================================")
        print(f"  File/Source   : {self.file_path}")
        print(f"  Total Rows    : {self.df.shape[0]:,}")
        print(f"  Total Columns : {self.df.shape[1]:,}")
        print(f"  Memory Usage  : {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"  Duplicate Rows: {self.df.duplicated().sum():,}")
        print(f"  Numeric Cols  : {len(self.num_cols)} -> {self.num_cols}")
        print(f"  Categoric Cols: {len(self.cat_cols)} -> {self.cat_cols}\n")

        # Column data type & missing summary table
        summary = pd.DataFrame({
            "Data Type": self.df.dtypes,
            "Non-Null Count": self.df.notnull().sum(),
            "Null Count": self.df.isnull().sum(),
            "Null %": (self.df.isnull().sum() / len(self.df) * 100).round(2),
            "Unique Values": self.df.nunique()
        })
        return summary

    def numerical_summary(self) -> pd.DataFrame:
        """Computes extended summary statistics for numeric variables."""
        if not self.num_cols:
            print("No numerical columns found.")
            return pd.DataFrame()

        desc = self.df[self.num_cols].describe().T
        desc["skewness"] = self.df[self.num_cols].skew()
        desc["kurtosis"] = self.df[self.num_cols].kurtosis()
        desc["iqr"] = desc["75%"] - desc["25%"]
        return desc.round(3)

    def categorical_summary(self) -> pd.DataFrame:
        """Computes breakdown of categorical variables."""
        if not self.cat_cols:
            print("No categorical columns found.")
            return pd.DataFrame()

        records = []
        for col in self.cat_cols:
            val_counts = self.df[col].value_counts(dropna=False)
            top_val = val_counts.index[0] if len(val_counts) > 0 else None
            top_freq = val_counts.iloc[0] if len(val_counts) > 0 else 0
            records.append({
                "Column": col,
                "Unique Values": self.df[col].nunique(),
                "Most Frequent": top_val,
                "Frequency": top_freq,
                "Top Category %": round((top_freq / len(self.df)) * 100, 2)
            })
        return pd.DataFrame(records).set_index("Column")

    def detect_outliers_iqr(self, threshold: float = 1.5) -> pd.DataFrame:
        """Identifies outlier counts per numeric feature based on 1.5 * IQR standard."""
        outlier_counts = {}
        for col in self.num_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_counts[col] = {
                "Outlier Count": len(outliers),
                "Outlier %": round((len(outliers) / len(self.df)) * 100, 2),
                "Lower Bound": round(lower_bound, 3),
                "Upper Bound": round(upper_bound, 3)
            }
        return pd.DataFrame.from_dict(outlier_counts, orient="index")

    def plot_missing_matrix(self, save_path: str = None):
        """Plots heatmap of missing value patterns."""
        null_counts = self.df.isnull().sum()
        if null_counts.sum() == 0:
            print("No missing values in dataset!")
            return

        plt.figure(figsize=(10, 4))
        sns.heatmap(self.df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
        plt.title(f"Missing Value Map - {self.name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def plot_numerical_distributions(self, output_dir: str = "."):
        """Plots histograms & KDE for each numerical column."""
        for col in self.num_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram + KDE
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[0], color="#4C72B0")
            axes[0].set_title(f"Distribution of {col}", fontweight="bold")
            
            # Boxplot
            sns.boxplot(x=self.df[col].dropna(), ax=axes[1], color="#55A868")
            axes[1].set_title(f"Boxplot of {col}", fontweight="bold")

            plt.tight_layout()
            file_name = os.path.join(output_dir, f"dist_{col}.png")
            plt.savefig(file_name)
            plt.close()
            print(f"Generated plot: {file_name}")

    def plot_correlation_matrix(self, save_path: str = "correlation_matrix.png"):
        """Computes and plots Pearson correlation matrix heatmap for numeric features."""
        if len(self.num_cols) < 2:
            print("Need at least 2 numerical columns for correlation analysis.")
            return

        corr = self.df[self.num_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
        plt.title(f"Correlation Matrix - {self.name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Generated correlation heatmap: {save_path}")

    def run_full_eda(self, output_dir: str = "eda_results"):
        """Executes full automated EDA pipeline and saves summaries and plots."""
        os.makedirs(output_dir, exist_ok=True)
        print("Starting Exploratory Data Analysis Pipeline...\n")
        
        # 1. Overview
        overview_df = self.overview()
        overview_df.to_csv(os.path.join(output_dir, "column_overview.csv"))

        # 2. Numerical Summary
        num_summary = self.numerical_summary()
        if not num_summary.empty:
            num_summary.to_csv(os.path.join(output_dir, "numerical_summary.csv"))
            print("Numerical Summary:")
            print(num_summary)
            print()

        # 3. Categorical Summary
        cat_summary = self.categorical_summary()
        if not cat_summary.empty:
            cat_summary.to_csv(os.path.join(output_dir, "categorical_summary.csv"))
            print("Categorical Summary:")
            print(cat_summary)
            print()

        # 4. Outlier Summary
        outliers_df = self.detect_outliers_iqr()
        if not outliers_df.empty:
            outliers_df.to_csv(os.path.join(output_dir, "outliers_summary.csv"))
            print("Outlier Detection (IQR Method):")
            print(outliers_df)
            print()

        # 4. Plots
        self.plot_numerical_distributions(output_dir=output_dir)
        self.plot_correlation_matrix(save_path=os.path.join(output_dir, "correlation_matrix.png"))
        
        print(f"\n[SUCCESS] EDA Complete! All results saved in folder: '{output_dir}'")

if __name__ == "__main__":
    import sys
    
    # Specify your dataset file path here or pass it via command line:
    # Example: python eda.py eda_processed_data.csv
    file_path = sys.argv[1] if len(sys.argv) > 1 else "eda_processed_data.csv"

    if not os.path.exists(file_path):
        print(f"Error: Dataset file '{file_path}' not found.")
        print("Usage: python eda.py <path_to_your_dataset.csv>")
        sys.exit(1)

    print(f"Loading dataset: '{file_path}'...\n")
    eda = ExploratoryDataAnalysis(file_path, name=os.path.basename(file_path))
    eda.run_full_eda(output_dir="eda_output")
