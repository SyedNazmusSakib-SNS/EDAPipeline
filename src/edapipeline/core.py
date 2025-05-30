import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest
import itertools 

class EDAPipeline:
    # --- Configuration ---
    HIGH_CARDINALITY_THRESHOLD = 50
    MEDIUM_CARDINALITY_THRESHOLD = 25
    TOP_N_CATEGORIES = 15 # For medium cardinality plots
    TARGET_CARDINALITY_THRESHOLD = 10 # Max unique values in target for hue

    def __init__(self, df, numerical_cols=None, categorical_cols=None, datetime_cols=None, target_col=None):
        self.output_dir = "eda_results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = df.copy() # Work on a copy to avoid modifying original df
        self.target_col = target_col

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
        print("\n=== 1. Dataset Overview ===")
        print(f"\nDataset Shape: {self.df.shape}")

        print("\nColumn Types:")
        # Display types clearly
        print(pd.DataFrame(self.df.dtypes, columns=['DataType']))

        print("\nIdentified Feature Types:")
        print(f"- Numerical:   {self.numerical_cols}")
        print(f"- Categorical: {self.categorical_cols}")
        print(f"- DateTime:    {self.datetime_cols}")
        print(f"- Target:      {self.target_col}")

        print("\nMissing Values (Top 10):")
        missing_counts = self.df.isnull().sum()
        missing_perc = (missing_counts / len(self.df)) * 100
        missing_df = pd.DataFrame({'Count': missing_counts, 'Percentage': missing_perc})
        print(missing_df[missing_df['Count'] > 0].sort_values('Percentage', ascending=False).head(10))

        print(f"\nNumber of duplicate rows: {self.df.duplicated().sum()}")

        # Memory usage
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"\nMemory Usage: {memory_usage:.2f} MB")

        print("\nSample Data (First 5 Rows):")
        print(self.df.head())

    def missing_value_analysis(self, figsize=(12, 6)):
        print("\n=== 2. Missing Value Analysis ===")
        missing_counts = self.df.isnull().sum()
        missing_df = pd.DataFrame({
            'Missing Values': missing_counts,
            'Percentage': (missing_counts / len(self.df)) * 100
        }).sort_values('Percentage', ascending=False)

        print("\nFull Missing Value Report:")
        print(missing_df[missing_df['Missing Values'] > 0])

        # Visualize only if there are missing values
        if missing_counts.sum() > 0:
            plt.figure(figsize=figsize)
            sns.heatmap(self.df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
            plt.title('Missing Value Heatmap')
            filename = "missing_value_heatmap.png"
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path)
            print(f"Plot saved to {full_path}")
            plt.show()
        else:
            print("\nNo missing values found in the dataset.")

    def analyze_numerical_features(self, figsize=(15, 5)):
        print("\n=== 3. Univariate Analysis: Numerical Features ===")
        if not self.numerical_cols:
            print("No numerical features identified.")
            return

        for col in self.numerical_cols:
            print(f"\n--- Analysis for '{col}' ---")

            # Check if column is empty or all NaN
            if self.df[col].isnull().all():
                print(f"Skipping '{col}' as it contains only missing values.")
                continue
            if self.df[col].empty:
                 print(f"Skipping '{col}' as it is empty.")
                 continue

            # Statistical Summary
            print("\nStatistical Summary:")
            print(self.df[col].describe())
            # Add Median Absolute Deviation (Robust measure of spread)
            print(f"Median Absolute Deviation: {stats.median_abs_deviation(self.df[col].dropna()):.3f}")
            print(f"Skewness: {self.df[col].skew():.3f}")
            print(f"Kurtosis: {self.df[col].kurt():.3f}") # Fisher’s definition (normal==0)

            # Normality Test (using D'Agostino and Pearson's test)
            try:
                stat, p_value = normaltest(self.df[col].dropna())
                print(f"\nNormality Test (D'Agostino-Pearson) p-value: {p_value:.3f}")
                if p_value < 0.05:
                    print("  (Reject normality hypothesis H0 at alpha=0.05)")
                else:
                    print("  (Cannot reject normality hypothesis H0 at alpha=0.05)")
            except ValueError as e:
                 print(f"\nNormality Test could not be performed for '{col}': {e}")


            # Create a figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=figsize) # Reduced to 3 plots
            fig.suptitle(f"Distribution Analysis for '{col}'", fontsize=16)

            # 1. Histogram with KDE
            sns.histplot(data=self.df, x=col, kde=True, ax=axes[0])
            axes[0].set_title('Histogram & KDE')

            # 2. Box Plot (shows median, quartiles, outliers)
            sns.boxplot(y=self.df[col], ax=axes[1])
            axes[1].set_title('Box Plot')

            # 3. Q-Q Plot (comparison against normal distribution)
            stats.probplot(self.df[col].dropna(), dist="norm", plot=axes[2])
            axes[2].set_title('Q-Q Plot (vs Normal)')
            # Improve Q-Q plot labels
            axes[2].set_xlabel("Theoretical Quantiles (Normal)")
            axes[2].set_ylabel("Sample Quantiles")


            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            filename = f"numerical_analysis_{col}.png"
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path)
            print(f"Plot saved to {full_path}")
            plt.show()

            # --- CDF Plot Removed as requested ---

    def analyze_categorical_features(self, figsize=(15, 5)):
        print("\n=== 4. Univariate Analysis: Categorical Features ===")
        if not self.categorical_cols:
            print("No categorical features identified.")
            return

        for col in self.categorical_cols:
            print(f"\n--- Analysis for '{col}' ---")

            n_unique = self.df[col].nunique()
            value_counts = self.df[col].value_counts()
            value_percentages = self.df[col].value_counts(normalize=True) * 100

            print(f"\nNumber of Unique Categories: {n_unique}")

            # Always print value counts (maybe top N for high cardinality)
            print("\nValue Counts (Top 10):")
            print(value_counts.head(10))
            print("\nValue Percentages (Top 10):")
            print(value_percentages.head(10).round(2).astype(str) + '%')


            # --- Visualization Logic based on Cardinality ---
            if n_unique == 0:
                 print("Column is empty. No plot generated.")
                 continue
            elif n_unique > self.HIGH_CARDINALITY_THRESHOLD:
                print(f"\n'{col}' has high cardinality ({n_unique} > {self.HIGH_CARDINALITY_THRESHOLD}). Skipping detailed plots.")
                # Optionally: Mention top N categories if needed, but already printed above.

            elif n_unique > self.MEDIUM_CARDINALITY_THRESHOLD:
                print(f"\n'{col}' has medium cardinality ({self.MEDIUM_CARDINALITY_THRESHOLD} < {n_unique} <= {self.HIGH_CARDINALITY_THRESHOLD}). Showing Top {self.TOP_N_CATEGORIES} categories.")
                plt.figure(figsize=(max(figsize[0]*0.7, 8), max(n_unique*0.3, 5))) # Adjust figure size
                # Use horizontal bar plot for better label readability
                top_n_counts = value_counts.head(self.TOP_N_CATEGORIES)
                sns.barplot(y=top_n_counts.index, x=top_n_counts.values, orient='h')
                plt.title(f'Top {self.TOP_N_CATEGORIES} Categories for {col}')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.tight_layout()
                filename = f"categorical_medium_cardinality_{col}.png"
                full_path = os.path.join(self.output_dir, filename)
                plt.savefig(full_path)
                print(f"Plot saved to {full_path}")
                plt.show()

            else: # Low cardinality (< MEDIUM_CARDINALITY_THRESHOLD)
                print(f"\n'{col}' has low cardinality ({n_unique} <= {self.MEDIUM_CARDINALITY_THRESHOLD}). Generating standard plots.")
                fig = plt.figure(figsize=figsize)
                fig.suptitle(f"Distribution Analysis for '{col}'", fontsize=16)

                # 1. Count Plot
                plt.subplot(1, 3, 1)
                sns.countplot(data=self.df, y=col, order=value_counts.index, orient='h') # Horizontal better for labels
                plt.title('Count Plot')
                plt.xlabel('Count')
                #plt.xticks(rotation=45, ha='right')

                # 2. Bar Plot of Percentages
                plt.subplot(1, 3, 2)
                value_percentages.plot(kind='barh') # Horizontal
                plt.title('Percentage Distribution')
                plt.xlabel('Percentage')
                plt.ylabel(col) # Ensure y-label is set
                #plt.xticks(rotation=45, ha='right')

                # 3. Pie Chart (Use only for very few categories, e.g., < 10)
                if n_unique <= 10:
                    plt.subplot(1, 3, 3)
                    plt.pie(value_percentages, labels=value_percentages.index, autopct='%1.1f%%', startangle=90, counterclock=False)
                    plt.title('Pie Chart')
                else:
                    # If too many categories for pie, maybe add another plot or leave empty
                    ax3 = fig.add_subplot(1, 3, 3)
                    ax3.text(0.5, 0.5, f'Pie chart skipped\n({n_unique} categories > 10)', horizontalalignment='center', verticalalignment='center', fontsize=12)
                    ax3.axis('off')


                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                filename = f"categorical_low_cardinality_{col}.png"
                full_path = os.path.join(self.output_dir, filename)
                plt.savefig(full_path)
                print(f"Plot saved to {full_path}")
                plt.show()

    def analyze_datetime_features(self, figsize=(15, 10)):
        print("\n=== 5. Univariate Analysis: DateTime Features ===")
        if not self.datetime_cols:
            print("No datetime features identified or converted.")
            return

        for col in self.datetime_cols:
            print(f"\n--- Analysis for '{col}' ---")

            if self.df[col].isnull().all():
                 print(f"Skipping '{col}' as it contains only missing values.")
                 continue

            print(f"\nTime Range: {self.df[col].min()} to {self.df[col].max()}")

            # --- Extract Time Components ---
            try:
                self.df[f'{col}_year'] = self.df[col].dt.year
                self.df[f'{col}_month'] = self.df[col].dt.month
                self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek # Monday=0, Sunday=6
                self.df[f'{col}_hour'] = self.df[col].dt.hour
                # Add more components if needed (e.g., week, quarter, dayofyear)
                temp_cols_created = True
            except AttributeError:
                print(f"Could not extract datetime components from '{col}'. Ensure it's a datetime type.")
                temp_cols_created = False
                continue # Skip plotting if components failed

            # --- Visualizations ---
            fig = plt.figure(figsize=figsize)
            fig.suptitle(f"DateTime Analysis for '{col}'", fontsize=16)

            # 1. Records over Years
            plt.subplot(2, 2, 1)
            sns.countplot(data=self.df, x=f'{col}_year')
            plt.title('Records per Year')
            plt.xticks(rotation=45)

            # 2. Records over Months
            plt.subplot(2, 2, 2)
            sns.countplot(data=self.df, x=f'{col}_month', palette='viridis')
            plt.title('Records per Month')
            plt.xticks(ticks=np.arange(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


            # 3. Records by Day of Week
            plt.subplot(2, 2, 3)
            sns.countplot(data=self.df, x=f'{col}_dayofweek', palette='magma')
            plt.title('Records by Day of Week')
            plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

            # 4. Records by Hour of Day
            plt.subplot(2, 2, 4)
            sns.countplot(data=self.df, x=f'{col}_hour', palette='plasma')
            plt.title('Records by Hour of Day')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename = f"datetime_analysis_{col}.png"
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path)
            print(f"Plot saved to {full_path}")
            plt.show()

            # --- Plot against Target Variable (if applicable) ---
            if self.target_col and self.target_col in self.numerical_cols:
                 print(f"\n--- Analyzing Numerical Target '{self.target_col}' against '{col}' Components ---")
                 fig_target = plt.figure(figsize=figsize)
                 fig_target.suptitle(f"Numerical Target '{self.target_col}' vs '{col}' Components", fontsize=16)

                 # Plot mean target value per time component
                 plt.subplot(2, 2, 1)
                 self.df.groupby(f'{col}_year')[self.target_col].mean().plot(kind='line', marker='o')
                 plt.title(f'Avg {self.target_col} per Year')
                 plt.ylabel(f'Average {self.target_col}')

                 plt.subplot(2, 2, 2)
                 self.df.groupby(f'{col}_month')[self.target_col].mean().plot(kind='line', marker='o')
                 plt.title(f'Avg {self.target_col} per Month')
                 plt.xticks(ticks=np.arange(1,13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                 plt.ylabel(f'Average {self.target_col}')

                 plt.subplot(2, 2, 3)
                 self.df.groupby(f'{col}_dayofweek')[self.target_col].mean().plot(kind='line', marker='o')
                 plt.title(f'Avg {self.target_col} per Day of Week')
                 plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                 plt.ylabel(f'Average {self.target_col}')

                 plt.subplot(2, 2, 4)
                 self.df.groupby(f'{col}_hour')[self.target_col].mean().plot(kind='line', marker='o')
                 plt.title(f'Avg {self.target_col} per Hour')
                 plt.ylabel(f'Average {self.target_col}')

                 plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                 filename = f"datetime_vs_target_{self.target_col}_by_{col}.png"
                 full_path = os.path.join(self.output_dir, filename)
                 plt.savefig(full_path)
                 print(f"Plot saved to {full_path}")
                 plt.show()

            # --- Clean up temporary columns ---
            # It's generally better *not* to modify the df permanently inside a method.
            # However, if keeping them is desired, remove this cleanup.
            if temp_cols_created:
                try:
                    self.df.drop(columns=[f'{col}_year', f'{col}_month', f'{col}_dayofweek', f'{col}_hour'], inplace=True, errors='ignore')
                except Exception as e:
                    print(f"Warning: Could not drop temporary datetime columns for {col}: {e}")


    def correlation_analysis(self, figsize=(12, 8)):
        print("\n=== 6. Correlation Analysis (Numerical Features) ===")
        if len(self.numerical_cols) < 2:
            print("Need at least two numerical features for correlation analysis.")
            return

        # Include target if it's numerical for correlation context
        cols_to_correlate = self.numerical_cols.copy()
        if self.target_col and self.target_col in self.df.select_dtypes(include=np.number).columns:
             if self.target_col not in cols_to_correlate:
                 cols_to_correlate.append(self.target_col)

        if len(cols_to_correlate) < 2:
             print("Need at least two numerical features (including target, if numerical) for correlation.")
             return

        print(f"\nCalculating correlation for: {cols_to_correlate}")
        corr_matrix = self.df[cols_to_correlate].corr()

        # Correlation heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, center=0)
        plt.title('Correlation Matrix Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        filename = "correlation_heatmap.png"
        full_path = os.path.join(self.output_dir, filename)
        plt.savefig(full_path)
        print(f"Plot saved to {full_path}")
        plt.show()

        # Display correlations with the target variable (if defined and numerical)
        if self.target_col and self.target_col in corr_matrix.columns:
            print(f"\nCorrelations with Target Variable ('{self.target_col}'):")
            target_corr = corr_matrix[self.target_col].drop(self.target_col).sort_values(ascending=False)
            print(target_corr)

        # Scatter plot matrix (Pairplot) - consider performance for many features
        num_features_for_pairplot = len(self.numerical_cols) # Only non-target features
        if num_features_for_pairplot <= 6 and num_features_for_pairplot >= 2:  # Limit pairplot complexity
            print(f"\nGenerating Pair Plot for {num_features_for_pairplot} numerical features...")
            pairplot_hue = None
            if self.target_col and self.target_col in self.categorical_cols:
                 if self.df[self.target_col].nunique() < self.TARGET_CARDINALITY_THRESHOLD:
                      pairplot_hue = self.target_col
                      print(f"(Coloring by target '{self.target_col}')")

            sns.pairplot(self.df[self.numerical_cols + ([self.target_col] if pairplot_hue else [])], # Include target only if used for hue
                         hue=pairplot_hue,
                         diag_kind='kde', # Show density plots on diagonal
                         plot_kws={'alpha': 0.6}) # Make points slightly transparent
            plt.suptitle('Pair Plot of Numerical Features', y=1.02) # Adjust title position
            filename = "correlation_pairplot.png"
            full_path = os.path.join(self.output_dir, filename)
            # Save before show, pairplot creates its own figure
            plt.savefig(full_path)
            print(f"Plot saved to {full_path}")
            plt.show()
        elif num_features_for_pairplot > 6:
            print("\nSkipping pair plot due to high number of numerical features (> 6). Consider `numerical_bivariate_analysis`.")


    def categorical_bivariate_analysis(self, figsize=(10, 6)):
        print("\n=== 7. Bivariate Analysis: Numerical vs. Categorical ===")
        if not self.numerical_cols or not self.categorical_cols:
            print("Requires both numerical and categorical features.")
            return

        for num_col in self.numerical_cols:
            for cat_col in self.categorical_cols:

                # Skip if categorical column has too high cardinality for useful plots
                n_unique = self.df[cat_col].nunique()
                if n_unique > self.MEDIUM_CARDINALITY_THRESHOLD:
                     print(f"\nSkipping bivariate plot for {num_col} vs {cat_col} (categorical cardinality {n_unique} > {self.MEDIUM_CARDINALITY_THRESHOLD}).")
                     continue

                print(f"\n--- Analyzing '{num_col}' vs '{cat_col}' ---")
                fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1])) # Two plots side-by-side
                fig.suptitle(f"'{num_col}' distributed by '{cat_col}'", fontsize=16)

                # Box plot
                sns.boxplot(x=self.df[cat_col], y=self.df[num_col], ax=axes[0])
                axes[0].set_title('Box Plot')
                axes[0].tick_params(axis='x', rotation=45)

                # Violin plot (shows density)
                sns.violinplot(x=self.df[cat_col], y=self.df[num_col], ax=axes[1])
                axes[1].set_title('Violin Plot')
                axes[1].tick_params(axis='x', rotation=45)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                filename = f"categorical_bivariate_{num_col}_vs_{cat_col}.png"
                full_path = os.path.join(self.output_dir, filename)
                plt.savefig(full_path)
                print(f"Plot saved to {full_path}")
                plt.show()

    def numerical_bivariate_analysis(self, figsize=(8, 8)):
         print("\n=== 8. Bivariate Analysis: Numerical vs. Numerical ===")
         if len(self.numerical_cols) < 2:
              print("Need at least two numerical features for this analysis.")
              return

         # Use combinations to avoid duplicate pairs (e.g., A vs B and B vs A)
         plotted_pairs = set()
         for col1, col2 in itertools.combinations(self.numerical_cols, 2):
             # Check if pair (in any order) has been plotted
             pair = tuple(sorted((col1, col2)))
             if pair in plotted_pairs:
                 continue
             plotted_pairs.add(pair)

             print(f"\n--- Analyzing '{col1}' vs '{col2}' ---")

             hue_col = None
             plot_title = f"Joint Plot: '{col1}' vs '{col2}'"
             # Use target for hue if it's categorical and has low cardinality
             if self.target_col and self.target_col in self.categorical_cols:
                  if self.df[self.target_col].nunique() < self.TARGET_CARDINALITY_THRESHOLD:
                      hue_col = self.target_col
                      plot_title += f" (Colored by '{self.target_col}')"

             # Jointplot shows scatter and marginal distributions
             # Use 'reg' for regression line, 'kde' for density, or 'hist'
             try:
                 # Create jointplot with regression line and KDE marginals
                 # Using `jointplot` directly handles figure creation
                 g = sns.jointplot(data=self.df, x=col1, y=col2, hue=hue_col, kind='scatter', # Use scatter first
                                   height=figsize[0]*0.8) # Adjust height
                 # If scatter is too dense, consider 'hex' or 'kde' kind
                 # g = sns.jointplot(data=self.df, x=col1, y=col2, kind='hex', height=figsize[0]*0.8)

                 g.fig.suptitle(plot_title, y=1.02) # Add title above the plot

                 # Add correlation coefficient to the plot (optional)
                 try:
                     corr, _ = stats.pearsonr(self.df[col1].dropna(), self.df[col2].dropna())
                     g.ax_joint.text(0.1, 0.9, f'Pearson r = {corr:.2f}', transform=g.ax_joint.transAxes)
                 except ValueError:
                      pass # Handle cases with insufficient data for correlation

                 # tight_layout for jointplot is often tricky, g.fig.tight_layout() can be used
                 # plt.tight_layout() # May not work as expected with jointplot
                 g.fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Apply to the figure of jointplot
                 filename = f"numerical_bivariate_{col1}_vs_{col2}.png"
                 full_path = os.path.join(self.output_dir, filename)
                 # Save before show, jointplot creates its own figure
                 plt.savefig(full_path)
                 print(f"Plot saved to {full_path}")
                 plt.show()
             except Exception as e:
                 print(f"Could not generate joint plot for '{col1}' vs '{col2}'. Error: {e}")


    def detect_outliers(self, method='iqr', threshold=3.0):
        """Detects outliers using Z-score or IQR method."""
        print(f"\n=== 9. Outlier Analysis ({method.upper()} Method) ===")
        outlier_stats = {}

        if not self.numerical_cols:
            print("No numerical features to analyze for outliers.")
            return

        for col in self.numerical_cols:
            # Drop NA values for outlier calculations
            col_data = self.df[col].dropna()
            if col_data.empty:
                print(f"\nSkipping outlier detection for '{col}' (all values are NaN).")
                continue

            n_total = len(self.df) # Use total rows for percentage calculation

            if method.lower() == 'zscore':
                if col_data.std() == 0: # Handle zero standard deviation case
                     outliers = 0
                     print(f"\nWarning: Cannot calculate Z-scores for '{col}' (standard deviation is zero).")
                else:
                     z_scores = np.abs(stats.zscore(col_data))
                     outliers = (z_scores > threshold).sum()
                outlier_stats[col] = {
                    'Method': 'Z-score',
                    'Threshold': threshold,
                    'Num Outliers': outliers,
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
                    'Lower Bound': f"{lower_bound:.2f}",
                    'Upper Bound': f"{upper_bound:.2f}",
                    'Num Outliers': outliers,
                    'Percentage': (outliers / n_total) * 100
                }
            else:
                print(f"Unknown outlier detection method: {method}. Use 'zscore' or 'iqr'.")
                return # Exit if method is invalid

        # Print summary table
        outlier_df = pd.DataFrame.from_dict(outlier_stats, orient='index')
        print("\nOutlier Detection Summary:")
        print(outlier_df[outlier_df['Num Outliers'] > 0].sort_values('Percentage', ascending=False))


    def run_complete_analysis(self, outlier_method='iqr'):
        """Runs the full EDA pipeline."""
        print("="*50)
        print(" Starting Complete EDA Pipeline ".center(50, "="))
        print("="*50)

        self.data_overview()
        self.missing_value_analysis()
        self.analyze_numerical_features()
        self.analyze_categorical_features()
        self.analyze_datetime_features()
        self.correlation_analysis()
        self.categorical_bivariate_analysis() 
        self.numerical_bivariate_analysis()   
        self.detect_outliers(method=outlier_method) 

        print("\n" + "="*50)
        print(" EDA Pipeline Completed! ".center(50, "="))
        print("="*50)




# 2. Instantiate and Run Pipeline
# eda_pipeline = EDAPipeline(df=df, target_col='Soil_Erosion')
# eda_pipeline.run_complete_analysis(outlier_method='iqr') # Use IQR for outliers

# --- To run only specific parts ---
# eda_pipeline.data_overview()
# eda_pipeline.analyze_categorical_features()
# eda_pipeline.analyze_datetime_features()
# eda_pipeline.numerical_bivariate_analysis()