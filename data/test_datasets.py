import unittest

import pandas as pd

# Local imports 
from data.utils import load_dataset

"""
NOTE: There is a lot of repeated code here, but it's easier to write separate tests for different datasets,
even though that means rewriting the same test over and over.
"""

class TestDataset(unittest.TestCase):

    def test_kddcup99(self):

        X, y = load_dataset('kddcup99', use_subsample=True, subsample_size=0.1)
        
        # Check types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        
        # Check shapes
        assert X.shape[0] == y.shape[0]
        
        # Check that the benign traffic is labeled "normal"
        assert len(y == 'normal') > 0


    def test_cicids2017(self):

        X, y = load_dataset('cicids2017', use_subsample=True, subsample_size=0.1)
        
        # Check types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        
        # Check shapes
        assert X.shape[0] == y.shape[0]
        
        # Check that the benign traffic is labeled "normal"
        assert len(y == 'normal') > 0


    def test_nslkdd(self):

        X, y = load_dataset('nslkdd', use_subsample=True, subsample_size=0.1)
        
        # Check types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        
        # Check shapes
        assert X.shape[0] == y.shape[0]
        
        # Check that the benign traffic is labeled "normal"
        assert len(y == 'normal') > 0



if __name__ == '__main__':
    unittest.main()