# EDAPipeline: Automated Exploratory Data Analysis Toolkit

**EDAPipeline** is a comprehensive and automated Python toolkit designed to streamline and simplify the Exploratory Data Analysis (EDA) process. This library helps data scientists, analysts, and engineers efficiently understand data distributions, detect outliers, visualize relationships, and uncover meaningful insights from datasets—all with minimal code.

## Key Features

- **Automatic Data Type Detection**: Automatically categorizes features into numerical, categorical, and datetime types.
- **Comprehensive Data Overview**: Quickly summarize dataset shape, data types, missing values, and memory usage.
- **Missing Value Analysis**: Visualizes and reports missing data, highlighting areas needing attention.
- **Advanced Univariate Analysis**:
  - **Numerical Features**: Statistical summaries, normality tests, histograms, KDE plots, box plots, and Q-Q plots.
  - **Categorical Features**: Counts, percentages, bar plots, and pie charts for clear categorical distribution analysis.
  - **Datetime Features**: Time-series component analysis, including trends across years, months, weekdays, and hourly distributions.
- **Correlation Analysis**: Provides correlation heatmaps and pair plots to uncover relationships between numerical features.
- **Robust Bivariate Analysis**: Detailed plots and analyses for numerical-numerical and numerical-categorical feature interactions.
- **Outlier Detection**: Implements Z-score and Interquartile Range (IQR) methods to identify outliers effectively.

## Installation

Install `EDAPipeline` via pip:

```bash
pip install edapipeline
```

## Usage

Here is a basic example demonstrating how to quickly set up and run a complete EDA analysis:

```python
import pandas as pd
from edapipeline import EDAPipeline

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Initialize the pipeline
eda = EDAPipeline(df=df, target_col='target_variable')

# Run the complete EDA analysis
eda.run_complete_analysis(outlier_method='iqr')
```

## Selective Analysis

You can also perform selective analyses based on your needs:

```python
# Overview of dataset
eda.data_overview()

# Numerical feature analysis
eda.analyze_numerical_features()

# Categorical feature analysis
eda.analyze_categorical_features()

# Datetime analysis
eda.analyze_datetime_features()

# Correlation analysis
eda.correlation_analysis()
```

## Advanced Customization

The pipeline provides flexibility to configure various thresholds and parameters:

```python
eda.HIGH_CARDINALITY_THRESHOLD = 100
eda.TOP_N_CATEGORIES = 10
eda.detect_outliers(method='zscore', threshold=2.5)
```

## Dependencies

- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy

Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scipy
```

## Contributions

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests on GitHub.

## License

`EDAPipeline` is open-source and available under the MIT License.

---

Explore your data effortlessly with **EDAPipeline**—turning data into actionable insights.

