import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = MinMaxScaler()
        self.categorical_cols = None
        self.numerical_cols = None
        self.ohe_feature_names = None

    def fit(self, X, y=None):
        """Identify column types and fit the encoders."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a Pandas DataFrame")

        # Identify categorical and numerical columns
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Fit encoders
        if self.categorical_cols:
            self.ohe.fit(X[self.categorical_cols])
            self.ohe_feature_names = self.ohe.get_feature_names_out(self.categorical_cols)
        
        if self.numerical_cols:
            self.scaler.fit(X[self.numerical_cols])

        return self

    def transform(self, X):
        """Transform categorical and numerical features."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a Pandas DataFrame")

        transformed_data = []

        # One-hot encoding
        if self.categorical_cols:
            cat_transformed = self.ohe.transform(X[self.categorical_cols])
            cat_transformed_df = pd.DataFrame(cat_transformed, columns=self.ohe_feature_names, index=X.index)
            transformed_data.append(cat_transformed_df)

        # MinMax scaling
        if self.numerical_cols:
            num_transformed = self.scaler.transform(X[self.numerical_cols])
            num_transformed_df = pd.DataFrame(num_transformed, columns=self.numerical_cols, index=X.index)
            transformed_data.append(num_transformed_df)

        # Concatenate transformed data
        return pd.concat(transformed_data, axis=1)

    def fit_transform(self, X, y=None):
        """Fit and transform the data in one step."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X_transformed):
        """Inverse transform the encoded and scaled data back to the original format."""
        original_data = []

        # Inverse One-Hot Encoding
        if self.categorical_cols:
            cat_transformed = X_transformed[self.ohe_feature_names].values
            cat_inverse = self.ohe.inverse_transform(cat_transformed)
            cat_inverse_df = pd.DataFrame(cat_inverse, columns=self.categorical_cols, index=X_transformed.index)
            original_data.append(cat_inverse_df)

        # Inverse MinMax Scaling
        if self.numerical_cols:
            num_transformed = X_transformed[self.numerical_cols].values
            num_inverse = self.scaler.inverse_transform(num_transformed)
            num_inverse_df = pd.DataFrame(num_inverse, columns=self.numerical_cols, index=X_transformed.index)
            original_data.append(num_inverse_df)

        # Concatenate inverse transformed data
        return pd.concat(original_data, axis=1)


if __name__ == '__main__':

    data = {
        "col1": [1, 2, 3, 4, 5],
        "col2": ["a", "b", "c", "d", "e"],
        "col3": [10.0, 20.0, 30.0, 40.0, 50.0],
        "col4": ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
    }

    df = pd.DataFrame(data)

    preprocessor = Preprocessor()
    transformed_df = preprocessor.fit_transform(df)

    print(transformed_df)