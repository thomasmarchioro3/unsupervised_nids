import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from ._descriptor import CATEGORICAL_FEATURES, COLUMN_NAMES


class NSLKDD_Preprocessor:
    def __init__(self):
        """
        Preprocessor for NSL-KDD dataset used by Choi et al. (2019).
        """
        
        self.categorical_cols = CATEGORICAL_FEATURES
        self.numerical_cols = [col for col in COLUMN_NAMES if col not in CATEGORICAL_FEATURES+['class', 'label']]
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = MinMaxScaler()

        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a Pandas DataFrame")

        transformed_data = []

        if self.categorical_cols:
            self.encoder.fit(X[self.categorical_cols])
            self.encoder_feature_names = self.encoder.get_feature_names_out(self.categorical_cols)
        
        if self.numerical_cols:
            self.scaler.fit(X[self.numerical_cols])

        self.is_fitted = True

    
    def transform(self, X: pd.DataFrame):
        if not self.is_fitted:
            raise ValueError('Preprocessor is not fitted')

        transformed_data = []

        # One-hot encoding
        if self.categorical_cols:
            cat_transformed = self.encoder.transform(X[self.categorical_cols])
            cat_transformed_df = pd.DataFrame(
                cat_transformed, # type: ignore
                columns=self.encoder_feature_names, 
                index=X.index
            ) # type: ignore
            transformed_data.append(cat_transformed_df)

        # MinMax scaling
        if self.numerical_cols:
            num_transformed = self.scaler.transform(X[self.numerical_cols])
            num_transformed_df = pd.DataFrame(num_transformed, columns=self.numerical_cols, index=X.index)
            transformed_data.append(num_transformed_df)

        X = pd.concat(transformed_data, axis=1)

        return X
    
    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)
    