import pandas as pd

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

class NIDS_Preprocessor:

    def __init__(self, cat_cols: list=None):

        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        self.cat_cols = cat_cols
        self.is_fitted = False


    def fit(self, X: pd.DataFrame):
        X = X.copy()
        
        if self.cat_cols is None:
            self.cat_cols = self.get_cat_cols(X)

        X.loc[:, self.cat_cols] = self.encoder.fit_transform(X[self.cat_cols])
        self.scaler.fit(X)
        self.is_fitted = True
    
    def transform(self, X: pd.DataFrame):
        if not self.is_fitted:
            raise ValueError('Preprocessor is not fitted')

        X = X.copy()
        X.loc[:, self.cat_cols] = self.encoder.transform(X[self.cat_cols])
        X.loc[:] = self.scaler.transform(X)
        return X
    
    def fit_transform(self, X: pd.DataFrame):
        """
        NOTE: Redundant method, used to avoid copying the input DataFrame twice.
        """

        X = X.copy()
        
        if self.cat_cols is None:
            self.cat_cols = self.get_cat_cols(X)

        X.loc[:, self.cat_cols] = self.encoder.fit_transform(X[self.cat_cols])
        X.loc[:] = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X

    @staticmethod
    def get_cat_cols(X: pd.DataFrame) -> list:
        num_cols = X.select_dtypes('number').columns.tolist()
        cat_cols = list(set(X.columns) - set(num_cols))
        return cat_cols
    