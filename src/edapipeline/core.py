import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest
import itertools 
import os
from datetime import datetime
from pathlib import Path

class EDAPipeline:
    # --- Configuration ---
    HIGH_CARDINALITY_THRESHOLD = 50
    MEDIUM_CARDINALITY_THRESHOLD = 25
    TOP_N_CATEGORIES = 15 # For medium cardinality plots
    TARGET_CARDINALITY_THRESHOLD = 10 # Max unique values in target for hue

    def __init__(self, df, numerical_cols=None, categorical_cols=None, datetime_cols=None, target_col=None,
                 save_outputs=False, output_dir='./eda_outputs'):
        self.df = df.copy() # Work on a copy to avoid modifying original df
        self.target_col = target_col
        self.save_outputs = save_outputs
        self.output_dir = output_dir
        
        # Initialize metrics log
        self.metrics_log = []
        self.plot_counter = 0
        
        # Create output directory if saving is enabled
        if self.save_outputs:
            self.plots_dir = Path(output_dir) / 'plots'
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamp for this analysis run
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.metrics_file = Path(output_dir) / f'eda_metrics_report_{self.run_timestamp}.txt'
            
            # Initialize the metrics file with header
            self._write_log("="*80)
            self._write_log(f"EDA METRICS REPORT")
            self._write_log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._write_log("="*80 + "\n")

        # Identify column types if not provided
        self.numerical_cols = numerical_cols if numerical_cols else self._identify_numerical_cols()
        self.categorical_cols = categorical_cols if categorical_cols else self._identify_categorical_cols()
        self.datetime_cols = datetime_cols if datetime_cols else self._identify_datetime_cols()

        # Remove target column from feature lists if present
        if self.target_col:
            self.numerical_cols = [col for col in self.numerical_cols if col != self.target_col]
            self.categorical_cols = [col for col in self.categorical_cols if col != self.target_col]
            self.datetime_cols = [col for col in self.datetime_cols if col != self.target_col]

        # Set style for all plots
        sns.set_theme(style="whitegrid") # Use a clean seaborn theme
        sns.set_palette("husl")

    def _write_log(self, text):
        """Write text to metrics log file and print to console."""
        print(text)
        if self.save_outputs:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(str(text) + '\n')
    
    def _save_plot(self, fig, plot_name):
        """Save the current plot to the plots directory."""
        if self.save_outputs:
            self.plot_counter += 1
            filename = f"{self.plot_counter:03d}_{plot_name}.png"
            filepath = self.plots_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self._write_log(f"  → Plot saved: {filename}")

    def _identify_numerical_cols(self):
        # Exclude boolean types often treated as categorical
        return self.df.select_dtypes(include=np.number, exclude='bool').columns.tolist()

    def _identify_categorical_cols(self):
        # Include 'category', 'object', and 'bool' types
        return self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    def _identify_datetime_cols(self):
        # Convert potential datetime columns first
        for col in self.df.select_dtypes(include=['object']).columns:
             try:
                 # Attempt conversion, but be robust to errors if it's not a date
                 self.df[col] = pd.to_datetime(self.df[col], errors='ignore')
             except Exception:
                 pass # Ignore if conversion fails
        # Now select actual datetime columns
        return self.df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    def data_overview(self):
        self._write_log("\n" + "="*80)
        self._write_log("1. DATASET OVERVIEW")
        self._write_log("="*80)
        
        self._write_log(f"\nDataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")

        self._write_log("\nColumn Data Types:")
        dtype_info = pd.DataFrame(self.df.dtypes, columns=['DataType'])
        self._write_log(dtype_info.to_string())

        self._write_log("\nIdentified Feature Types:")
        self._write_log(f"  • Numerical Columns ({len(self.numerical_cols)}): {self.numerical_cols}")
        self._write_log(f"  • Categorical Columns ({len(self.categorical_cols)}): {self.categorical_cols}")
        self._write_log(f"  • DateTime Columns ({len(self.datetime_cols)}): {self.datetime_cols}")
        self._write_log(f"  • Target Column: {self.target_col}")

        self._write_log("\nMissing Values Analysis:")
        missing_counts = self.df.isnull().sum()
        missing_perc = (missing_counts / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_counts, 
            'Missing_Percentage': missing_perc
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        if len(missing_df) > 0:
            self._write_log(f"\nColumns with Missing Values (Top 10):")
            self._write_log(missing_df.head(10).to_string())
            self._write_log(f"\nTotal columns with missing values: {len(missing_df)}")
            self._write_log(f"Total missing cells: {missing_counts.sum()}")
        else:
            self._write_log("\n✓ No missing values found in the dataset")

        # Memory usage
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
        self._write_log(f"\nMemory Usage: {memory_usage:.2f} MB")

        self._write_log("\nSample Data (First 5 Rows):")
        self._write_log(self.df.head().to_string())

    def missing_value_analysis(self, figsize=(12, 6)):
        self._write_log("\n" + "="*80)
        self._write_log("2. MISSING VALUE ANALYSIS")
        self._write_log("="*80)
        
        missing_counts = self.df.isnull().sum()
        missing_df = pd.DataFrame({
            'Missing Values': missing_counts,
            'Percentage': (missing_counts / len(self.df)) * 100
        }).sort_values('Percentage', ascending=False)

        self._write_log("\nComplete Missing Value Report:")
        missing_with_values = missing_df[missing_df['Missing Values'] > 0]
        if len(missing_with_values) > 0:
            self._write_log(missing_with_values.to_string())
        else:
            self._write_log("✓ No missing values detected")

        # Visualize only if there are missing values
        if missing_counts.sum() > 0:
            fig = plt.figure(figsize=figsize)
            sns.heatmap(self.df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
            plt.title('Missing Value Heatmap')
            self._save_plot(fig, 'missing_values_heatmap')
            plt.show()
            plt.close()

    def analyze_numerical_features(self, figsize=(15, 5)):
        self._write_log("\n" + "="*80)
        self._write_log("3. UNIVARIATE ANALYSIS: NUMERICAL FEATURES")
        self._write_log("="*80)
        
        if not self.numerical_cols:
            self._write_log("\n✗ No numerical features identified")
            return

        for col in self.numerical_cols:
            self._write_log(f"\n{'─'*80}")
            self._write_log(f"NUMERICAL FEATURE: '{col}'")
            self._write_log(f"{'─'*80}")

            # Check if column is empty or all NaN
            if self.df[col].isnull().all():
                self._write_log(f"⚠ Skipping '{col}' - contains only missing values")
                continue
            if self.df[col].empty:
                self._write_log(f"⚠ Skipping '{col}' - column is empty")
                continue

            # Statistical Summary
            self._write_log("\n📊 Descriptive Statistics:")
            desc_stats = self.df[col].describe()
            self._write_log(desc_stats.to_string())
            
            # Additional robust statistics
            try:
                mad = stats.median_abs_deviation(self.df[col].dropna())
                skewness = self.df[col].skew()
                kurtosis = self.df[col].kurt()
                
                self._write_log(f"\n📈 Distribution Metrics:")
                self._write_log(f"  • Median Absolute Deviation (MAD): {mad:.4f}")
                self._write_log(f"  • Skewness: {skewness:.4f}")
                self._write_log(f"  • Kurtosis (Fisher): {kurtosis:.4f}")
                
                # Interpret skewness
                if abs(skewness) < 0.5:
                    skew_interp = "approximately symmetric"
                elif skewness > 0.5:
                    skew_interp = "right-skewed (positive skew)"
                else:
                    skew_interp = "left-skewed (negative skew)"
                self._write_log(f"    → Distribution is {skew_interp}")
                
                # Interpret kurtosis
                if abs(kurtosis) < 0.5:
                    kurt_interp = "mesokurtic (normal-like tails)"
                elif kurtosis > 0.5:
                    kurt_interp = "leptokurtic (heavy tails, more outliers)"
                else:
                    kurt_interp = "platykurtic (light tails, fewer outliers)"
                self._write_log(f"    → Distribution is {kurt_interp}")
            
            except Exception as e:
                self._write_log(f"⚠ Could not calculate distribution metrics: {e}")

            # Normality Test
            self._write_log(f"\n🔬 Normality Test (D'Agostino-Pearson):")
            try:
                stat, p_value = normaltest(self.df[col].dropna())
                self._write_log(f"  • Test Statistic: {stat:.4f}")
                self._write_log(f"  • P-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    self._write_log(f"  • Result: REJECT normality hypothesis (α=0.05)")
                    self._write_log(f"    → Data is NOT normally distributed")
                else:
                    self._write_log(f"  • Result: CANNOT REJECT normality hypothesis (α=0.05)")
                    self._write_log(f"    → Data may be normally distributed")
            except ValueError as e:
                self._write_log(f"  ⚠ Normality test could not be performed: {e}")

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            fig.suptitle(f"Distribution Analysis: '{col}'", fontsize=16)

            # 1. Histogram with KDE
            sns.histplot(data=self.df, x=col, kde=True, ax=axes[0])
            axes[0].set_title('Histogram & KDE')

            # 2. Box Plot
            sns.boxplot(y=self.df[col], ax=axes[1])
            axes[1].set_title('Box Plot')

            # 3. Q-Q Plot
            stats.probplot(self.df[col].dropna(), dist="norm", plot=axes[2])
            axes[2].set_title('Q-Q Plot (vs Normal)')
            axes[2].set_xlabel("Theoretical Quantiles")
            axes[2].set_ylabel("Sample Quantiles")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self._save_plot(fig, f'numerical_{col}')
            plt.show()
            plt.close()

    def analyze_categorical_features(self, figsize=(15, 5)):
        self._write_log("\n" + "="*80)
        self._write_log("4. UNIVARIATE ANALYSIS: CATEGORICAL FEATURES")
        self._write_log("="*80)
        
        if not self.categorical_cols:
            self._write_log("\n✗ No categorical features identified")
            return

        for col in self.categorical_cols:
            self._write_log(f"\n{'─'*80}")
            self._write_log(f"CATEGORICAL FEATURE: '{col}'")
            self._write_log(f"{'─'*80}")

            n_unique = self.df[col].nunique()
            value_counts = self.df[col].value_counts()
            value_percentages = self.df[col].value_counts(normalize=True) * 100

            self._write_log(f"\n📊 Cardinality Metrics:")
            self._write_log(f"  • Unique Categories: {n_unique}")
            self._write_log(f"  • Mode (Most Frequent): {value_counts.index[0] if len(value_counts) > 0 else 'N/A'}")
            self._write_log(f"  • Mode Frequency: {value_counts.iloc[0] if len(value_counts) > 0 else 0} ({value_percentages.iloc[0]:.2f}%)")

            # Category distribution
            self._write_log(f"\n📈 Value Distribution (Top 10):")
            top_10 = pd.DataFrame({
                'Category': value_counts.head(10).index,
                'Count': value_counts.head(10).values,
                'Percentage': value_percentages.head(10).values
            })
            self._write_log(top_10.to_string(index=False))

            # Entropy calculation (measure of randomness)
            try:
                entropy = stats.entropy(value_counts.values)
                max_entropy = np.log(n_unique)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                self._write_log(f"\n🎲 Information Theory Metrics:")
                self._write_log(f"  • Entropy: {entropy:.4f}")
                self._write_log(f"  • Normalized Entropy: {normalized_entropy:.4f}")
                self._write_log(f"    → {normalized_entropy*100:.1f}% of maximum possible entropy")
            except Exception:
                pass

            # Visualization Logic
            if n_unique == 0:
                self._write_log("\n⚠ Column is empty - no plot generated")
                continue
            elif n_unique > self.HIGH_CARDINALITY_THRESHOLD:
                self._write_log(f"\n⚠ High cardinality ({n_unique} > {self.HIGH_CARDINALITY_THRESHOLD}) - skipping detailed plots")

            elif n_unique > self.MEDIUM_CARDINALITY_THRESHOLD:
                self._write_log(f"\n📊 Medium cardinality - showing top {self.TOP_N_CATEGORIES} categories")
                fig = plt.figure(figsize=(max(figsize[0]*0.7, 8), max(n_unique*0.3, 5)))
                top_n_counts = value_counts.head(self.TOP_N_CATEGORIES)
                sns.barplot(y=top_n_counts.index, x=top_n_counts.values, orient='h')
                plt.title(f'Top {self.TOP_N_CATEGORIES} Categories: {col}')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.tight_layout()
                self._save_plot(fig, f'categorical_{col}_top{self.TOP_N_CATEGORIES}')
                plt.show()
                plt.close()

            else:
                self._write_log(f"\n📊 Low cardinality ({n_unique}) - generating complete visualizations")
                fig = plt.figure(figsize=figsize)
                fig.suptitle(f"Distribution Analysis: '{col}'", fontsize=16)

                # 1. Count Plot
                plt.subplot(1, 3, 1)
                sns.countplot(data=self.df, y=col, order=value_counts.index, orient='h')
                plt.title('Count Plot')
                plt.xlabel('Count')

                # 2. Percentage Bar Plot
                plt.subplot(1, 3, 2)
                value_percentages.plot(kind='barh')
                plt.title('Percentage Distribution')
                plt.xlabel('Percentage')
                plt.ylabel(col)

                # 3. Pie Chart (only for <= 10 categories)
                if n_unique <= 10:
                    plt.subplot(1, 3, 3)
                    plt.pie(value_percentages, labels=value_percentages.index, autopct='%1.1f%%', 
                           startangle=90, counterclock=False)
                    plt.title('Pie Chart')
                else:
                    ax3 = fig.add_subplot(1, 3, 3)
                    ax3.text(0.5, 0.5, f'Pie chart omitted\n({n_unique} categories)', 
                            ha='center', va='center', fontsize=12)
                    ax3.axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                self._save_plot(fig, f'categorical_{col}')
                plt.show()
                plt.close()

    def analyze_datetime_features(self, figsize=(15, 10)):
        self._write_log("\n" + "="*80)
        self._write_log("5. UNIVARIATE ANALYSIS: DATETIME FEATURES")
        self._write_log("="*80)
        
        if not self.datetime_cols:
            self._write_log("\n✗ No datetime features identified")
            return

        for col in self.datetime_cols:
            self._write_log(f"\n{'─'*80}")
            self._write_log(f"DATETIME FEATURE: '{col}'")
            self._write_log(f"{'─'*80}")

            if self.df[col].isnull().all():
                self._write_log(f"\n⚠ Skipping '{col}' - contains only missing values")
                continue

            min_date = self.df[col].min()
            max_date = self.df[col].max()
            date_range = max_date - min_date
            
            self._write_log(f"\n📅 Temporal Range:")
            self._write_log(f"  • Earliest Date: {min_date}")
            self._write_log(f"  • Latest Date: {max_date}")
            self._write_log(f"  • Time Span: {date_range.days} days ({date_range.days/365.25:.2f} years)")

            # Extract time components
            try:
                self.df[f'{col}_year'] = self.df[col].dt.year
                self.df[f'{col}_month'] = self.df[col].dt.month
                self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek
                self.df[f'{col}_hour'] = self.df[col].dt.hour
                temp_cols_created = True
                
                # Record counts
                self._write_log(f"\n📊 Temporal Distribution:")
                self._write_log(f"  • Unique Years: {self.df[f'{col}_year'].nunique()}")
                self._write_log(f"  • Unique Months: {self.df[f'{col}_month'].nunique()}")
                self._write_log(f"  • Records span {self.df[f'{col}_dayofweek'].nunique()} different days of week")
                
            except AttributeError:
                self._write_log(f"\n⚠ Could not extract datetime components from '{col}'")
                temp_cols_created = False
                continue

            # Visualizations
            fig = plt.figure(figsize=figsize)
            fig.suptitle(f"DateTime Analysis: '{col}'", fontsize=16)

            plt.subplot(2, 2, 1)
            sns.countplot(data=self.df, x=f'{col}_year')
            plt.title('Records per Year')
            plt.xticks(rotation=45)

            plt.subplot(2, 2, 2)
            sns.countplot(data=self.df, x=f'{col}_month', palette='viridis')
            plt.title('Records per Month')
            plt.xticks(ticks=np.arange(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

            plt.subplot(2, 2, 3)
            sns.countplot(data=self.df, x=f'{col}_dayofweek', palette='magma')
            plt.title('Records by Day of Week')
            plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

            plt.subplot(2, 2, 4)
            sns.countplot(data=self.df, x=f'{col}_hour', palette='plasma')
            plt.title('Records by Hour of Day')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self._save_plot(fig, f'datetime_{col}_distribution')
            plt.show()
            plt.close()

            # Target analysis if applicable
            if self.target_col and self.target_col in self.numerical_cols:
                self._write_log(f"\n🎯 Target Variable Analysis ('{self.target_col}' vs '{col}'):")
                
                fig_target = plt.figure(figsize=figsize)
                fig_target.suptitle(f"'{self.target_col}' vs '{col}' Components", fontsize=16)

                plt.subplot(2, 2, 1)
                yearly_avg = self.df.groupby(f'{col}_year')[self.target_col].mean()
                yearly_avg.plot(kind='line', marker='o')
                plt.title(f'Avg {self.target_col} per Year')
                plt.ylabel(f'Average {self.target_col}')
                self._write_log(f"  • Year with highest avg {self.target_col}: {yearly_avg.idxmax()} ({yearly_avg.max():.4f})")

                plt.subplot(2, 2, 2)
                monthly_avg = self.df.groupby(f'{col}_month')[self.target_col].mean()
                monthly_avg.plot(kind='line', marker='o')
                plt.title(f'Avg {self.target_col} per Month')
                plt.xticks(ticks=np.arange(1,13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                plt.ylabel(f'Average {self.target_col}')
                self._write_log(f"  • Month with highest avg {self.target_col}: {monthly_avg.idxmax()} ({monthly_avg.max():.4f})")

                plt.subplot(2, 2, 3)
                dow_avg = self.df.groupby(f'{col}_dayofweek')[self.target_col].mean()
                dow_avg.plot(kind='line', marker='o')
                plt.title(f'Avg {self.target_col} per Day of Week')
                plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                plt.ylabel(f'Average {self.target_col}')
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                self._write_log(f"  • Day with highest avg {self.target_col}: {days[dow_avg.idxmax()]} ({dow_avg.max():.4f})")

                plt.subplot(2, 2, 4)
                hourly_avg = self.df.groupby(f'{col}_hour')[self.target_col].mean()
                hourly_avg.plot(kind='line', marker='o')
                plt.title(f'Avg {self.target_col} per Hour')
                plt.ylabel(f'Average {self.target_col}')
                self._write_log(f"  • Hour with highest avg {self.target_col}: {hourly_avg.idxmax()}:00 ({hourly_avg.max():.4f})")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                self._save_plot(fig, f'datetime_{col}_vs_target')
                plt.show()
                plt.close()

            # Cleanup temporary columns
            if temp_cols_created:
                try:
                    self.df.drop(columns=[f'{col}_year', f'{col}_month', f'{col}_dayofweek', f'{col}_hour'], 
                               inplace=True, errors='ignore')
                except Exception as e:
                    self._write_log(f"⚠ Warning: Could not drop temporary columns: {e}")

    def correlation_analysis(self, figsize=(12, 8)):
        self._write_log("\n" + "="*80)
        self._write_log("6. CORRELATION ANALYSIS (NUMERICAL FEATURES)")
        self._write_log("="*80)
        
        if len(self.numerical_cols) < 2:
            self._write_log("\n✗ Need at least 2 numerical features for correlation analysis")
            return

        # Include target if numerical
        cols_to_correlate = self.numerical_cols.copy()
        if self.target_col and self.target_col in self.df.select_dtypes(include=np.number).columns:
            if self.target_col not in cols_to_correlate:
                cols_to_correlate.append(self.target_col)

        if len(cols_to_correlate) < 2:
            self._write_log("\n✗ Insufficient numerical features for correlation")
            return

        self._write_log(f"\n📊 Calculating correlations for {len(cols_to_correlate)} features")
        corr_matrix = self.df[cols_to_correlate].corr()

        # Find strongest correlations
        self._write_log(f"\n🔗 Strongest Positive Correlations (excluding diagonal):")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
        self._write_log(corr_df.head(10).to_string(index=False))
        
        self._write_log(f"\n🔗 Strongest Negative Correlations:")
        self._write_log(corr_df.tail(10).to_string(index=False))

        # Correlation heatmap
        fig = plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                   linewidths=.5, center=0)
        plt.title('Correlation Matrix Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        self._save_plot(fig, 'correlation_heatmap')
        plt.show()
        plt.close()

        # Target correlations
        if self.target_col and self.target_col in corr_matrix.columns:
            self._write_log(f"\n🎯 Correlations with Target Variable ('{self.target_col}'):")
            target_corr = corr_matrix[self.target_col].drop(self.target_col).sort_values(ascending=False)
            self._write_log(target_corr.to_string())
            
            # Identify features most correlated with target
            self._write_log(f"\n  • Strongest positive correlation: {target_corr.idxmax()} (r={target_corr.max():.4f})")
            self._write_log(f"  • Strongest negative correlation: {target_corr.idxmin()} (r={target_corr.min():.4f})")

        # Pairplot
        num_features_for_pairplot = len(self.numerical_cols)
        if 2 <= num_features_for_pairplot <= 6:
            self._write_log(f"\n📊 Generating pair plot for {num_features_for_pairplot} features...")
            pairplot_hue = None
            if self.target_col and self.target_col in self.categorical_cols:
                if self.df[self.target_col].nunique() < self.TARGET_CARDINALITY_THRESHOLD:
                    pairplot_hue = self.target_col

            g = sns.pairplot(self.df[self.numerical_cols + ([self.target_col] if pairplot_hue else [])],
                           hue=pairplot_hue, diag_kind='kde', plot_kws={'alpha': 0.6})
            plt.suptitle('Pair Plot of Numerical Features', y=1.02)
            self._save_plot(g.fig, 'pairplot_numerical')
            plt.show()
            plt.close()
        elif num_features_for_pairplot > 6:
            self._write_log(f"\n⚠ Skipping pair plot (too many features: {num_features_for_pairplot} > 6)")

    def categorical_bivariate_analysis(self, figsize=(10, 6)):
        self._write_log("\n" + "="*80)
        self._write_log("7. BIVARIATE ANALYSIS: NUMERICAL vs CATEGORICAL")
        self._write_log("="*80)
        
        if not self.numerical_cols or not self.categorical_cols:
            self._write_log("\n✗ Requires both numerical and categorical features")
            return

        analysis_count = 0
        for num_col in self.numerical_cols:
            for cat_col in self.categorical_cols:
                n_unique = self.df[cat_col].nunique()
                if n_unique > self.MEDIUM_CARDINALITY_THRESHOLD:
                    continue

                analysis_count += 1
                self._write_log(f"\n{'─'*80}")
                self._write_log(f"Analyzing: '{num_col}' vs '{cat_col}'")
                self._write_log(f"{'─'*80}")

                # Calculate statistics per category
                grouped = self.df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std', 'count'])
                self._write_log(f"\n📊 Statistics by Category:")
                self._write_log(grouped.to_string())

                fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
                fig.suptitle(f"'{num_col}' by '{cat_col}'", fontsize=16)

                sns.boxplot(x=self.df[cat_col], y=self.df[num_col], ax=axes[0])
                axes[0].set_title('Box Plot')
                axes[0].tick_params(axis='x', rotation=45)

                sns.violinplot(x=self.df[cat_col], y=self.df[num_col], ax=axes[1])
                axes[1].set_title('Violin Plot')
                axes[1].tick_params(axis='x', rotation=45)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                self._save_plot(fig, f'bivariate_{num_col}_vs_{cat_col}')
                plt.show()
                plt.close()
        
        self._write_log(f"\n✓ Completed {analysis_count} bivariate analyses")

    def numerical_bivariate_analysis(self, figsize=(8, 8)):
        self._write_log("\n" + "="*80)
        self._write_log("8. BIVARIATE ANALYSIS: NUMERICAL vs NUMERICAL")
        self._write_log("="*80)
        
        if len(self.numerical_cols) < 2:
            self._write_log("\n✗ Need at least 2 numerical features")
            return

        plotted_pairs = set()
        pair_count = 0
        
        for col1, col2 in itertools.combinations(self.numerical_cols, 2):
            pair = tuple(sorted((col1, col2)))
            if pair in plotted_pairs:
                continue
            plotted_pairs.add(pair)
            pair_count += 1

            self._write_log(f"\n{'─'*80}")
            self._write_log(f"Analyzing: '{col1}' vs '{col2}'")
            self._write_log(f"{'─'*80}")

            # Calculate correlation
            try:
                corr, p_value = stats.pearsonr(self.df[col1].dropna(), self.df[col2].dropna())
                self._write_log(f"\n🔗 Pearson Correlation:")
                self._write_log(f"  • Correlation coefficient (r): {corr:.4f}")
                self._write_log(f"  • P-value: {p_value:.4f}")
                
                if abs(corr) > 0.7:
                    strength = "Strong"
                elif abs(corr) > 0.4:
                    strength = "Moderate"
                elif abs(corr) > 0.2:
                    strength = "Weak"
                else:
                    strength = "Very Weak"
                
                direction = "positive" if corr > 0 else "negative"
                self._write_log(f"  • Interpretation: {strength} {direction} correlation")
                
                if p_value < 0.05:
                    self._write_log(f"  • Significance: Statistically significant (p < 0.05)")
                else:
                    self._write_log(f"  • Significance: Not statistically significant (p >= 0.05)")
                    
            except ValueError:
                self._write_log(f"\n⚠ Could not calculate correlation (insufficient data)")
                corr = None

            # Create jointplot
            hue_col = None
            if self.target_col and self.target_col in self.categorical_cols:
                if self.df[self.target_col].nunique() < self.TARGET_CARDINALITY_THRESHOLD:
                    hue_col = self.target_col

            try:
                g = sns.jointplot(data=self.df, x=col1, y=col2, hue=hue_col, 
                                kind='scatter', height=figsize[0]*0.8)
                g.fig.suptitle(f"Joint Plot: '{col1}' vs '{col2}'", y=1.02)
                
                if corr is not None:
                    g.ax_joint.text(0.1, 0.9, f'r = {corr:.3f}', 
                                  transform=g.ax_joint.transAxes, 
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                self._save_plot(g.fig, f'numerical_bivariate_{col1}_vs_{col2}')
                plt.show()
                plt.close()
            except Exception as e:
                self._write_log(f"\n⚠ Could not generate joint plot: {e}")
        
        self._write_log(f"\n✓ Completed {pair_count} numerical bivariate analyses")

    def detect_outliers(self, method='iqr', threshold=3.0):
        self._write_log("\n" + "="*80)
        self._write_log(f"9. OUTLIER ANALYSIS ({method.upper()} Method)")
        self._write_log("="*80)
        
        if not self.numerical_cols:
            self._write_log("\n✗ No numerical features to analyze for outliers")
            return

        outlier_stats = {}

        for col in self.numerical_cols:
            col_data = self.df[col].dropna()
            if col_data.empty:
                continue

            n_total = len(self.df)

            if method.lower() == 'zscore':
                if col_data.std() == 0:
                    outliers = 0
                    self._write_log(f"\n⚠ '{col}': Cannot calculate Z-scores (std=0)")
                else:
                    z_scores = np.abs(stats.zscore(col_data))
                    outliers = (z_scores > threshold).sum()
                outlier_stats[col] = {
                    'Method': 'Z-score',
                    'Threshold': threshold,
                    'Num_Outliers': outliers,
                    'Percentage': (outliers / n_total) * 100
                }

            elif method.lower() == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                outlier_stats[col] = {
                    'Method': 'IQR',
                    'Lower_Bound': f"{lower_bound:.3f}",
                    'Upper_Bound': f"{upper_bound:.3f}",
                    'Num_Outliers': outliers,
                    'Percentage': (outliers / n_total) * 100
                }
            else:
                self._write_log(f"\n✗ Unknown method: {method}. Use 'zscore' or 'iqr'")
                return

        # Create summary table
        outlier_df = pd.DataFrame.from_dict(outlier_stats, orient='index')
        outliers_found = outlier_df[outlier_df['Num_Outliers'] > 0].sort_values('Percentage', ascending=False)
        
        self._write_log(f"\n📊 Outlier Detection Summary:")
        if len(outliers_found) > 0:
            self._write_log(outliers_found.to_string())
            self._write_log(f"\n  • Total features with outliers: {len(outliers_found)}")
            self._write_log(f"  • Feature with most outliers: {outliers_found.index[0]} ({outliers_found.iloc[0]['Percentage']:.2f}%)")
        else:
            self._write_log("\n✓ No outliers detected in any numerical feature")

    def run_complete_analysis(self, outlier_method='iqr'):
        """Runs the full EDA pipeline."""
        self._write_log("\n" + "="*80)
        self._write_log("STARTING COMPLETE EDA PIPELINE")
        self._write_log("="*80)
        
        start_time = datetime.now()

        self.data_overview()
        self.missing_value_analysis()
        self.analyze_numerical_features()
        self.analyze_categorical_features()
        self.analyze_datetime_features()
        self.correlation_analysis()
        self.categorical_bivariate_analysis() 
        self.numerical_bivariate_analysis()   
        self.detect_outliers(method=outlier_method) 

        end_time = datetime.now()
        duration = end_time - start_time

        self._write_log("\n" + "="*80)
        self._write_log("EDA PIPELINE COMPLETED")
        self._write_log("="*80)
        self._write_log(f"\n⏱ Total execution time: {duration}")
        self._write_log(f"📊 Total plots generated: {self.plot_counter}")
        
        if self.save_outputs:
            self._write_log(f"\n💾 Output Location:")
            self._write_log(f"  • Plots directory: {self.plots_dir}")
            self._write_log(f"  • Metrics report: {self.metrics_file}")
            self._write_log(f"\n✓ All results saved successfully!")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
"""
# Load your data
import pandas as pd
df = pd.read_csv('your_data.csv')

# Option 1: Run complete analysis WITH saving outputs
eda = EDAPipeline(
    df=df, 
    target_col='your_target_column',
    save_outputs=True,
    output_dir='./eda_results'
)
eda.run_complete_analysis(outlier_method='iqr')

# Option 2: Run complete analysis WITHOUT saving (original behavior)
eda = EDAPipeline(df=df, target_col='your_target_column')
eda.run_complete_analysis()

# Option 3: Run only specific analyses with saving
eda = EDAPipeline(df=df, save_outputs=True, output_dir='./my_eda')
eda.data_overview()
eda.correlation_analysis()
eda.analyze_numerical_features()
"""