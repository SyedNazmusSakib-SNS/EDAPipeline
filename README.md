```markdown
# EDAPipeline

A comprehensive Exploratory Data Analysis (EDA) toolkit that streamlines the process of analyzing datasets through visualization and statistical methods.

## Features

- Automated data type detection (numerical, categorical, datetime)
- Comprehensive univariate analysis for all data types
- Correlation analysis with heatmaps
- Bivariate analysis between different feature types
- Datetime feature decomposition and analysis
- Outlier detection using multiple methods
- Customizable visualization options

## Installation

```bash
pip install edapipeline
```

## Quick Start

```python
from edapipeline import EDAPipeline
import pandas as pd

# Load your dataset
df = pd.read_csv('your_data.csv')

# Initialize the pipeline
eda = EDAPipeline(df, target_col='your_target_column')

# Run the complete analysis
eda.run_complete_analysis()

# Or run specific analyses
eda.data_overview()
eda.analyze_numerical_features()
eda.correlation_analysis()
```

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scipy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

### 2. Create a Basic __init__.py File

Edit your `src/edapipeline/__init__.py` file:

```python
"""EDAPipeline - Comprehensive EDA toolkit for data analysis."""

from .core import EDAPipeline
from .__version__ import __version__

__all__ = ['EDAPipeline']
```


