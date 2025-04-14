"""Tests for EDAPipeline."""

import unittest
import pandas as pd
import numpy as np
from edapipeline import EDAPipeline

class TestEDAPipeline(unittest.TestCase):
    """Tests for the EDAPipeline class."""
    
    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'numerical': np.random.rand(100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'date': pd.date_range('2023-01-01', periods=100)
        })
        self.pipeline = EDAPipeline(self.df)
    
    def test_initialization(self):
        """Test that the pipeline initializes correctly."""
        self.assertIsInstance(self.pipeline, EDAPipeline)
        self.assertEqual(len(self.pipeline.numerical_cols), 1)
        self.assertEqual(len(self.pipeline.categorical_cols), 1)
        self.assertEqual(len(self.pipeline.datetime_cols), 1)
    
    def test_data_overview(self):
        """Test data_overview method runs without errors."""
        try:
            self.pipeline.data_overview()
            success = True
        except:
            success = False
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()