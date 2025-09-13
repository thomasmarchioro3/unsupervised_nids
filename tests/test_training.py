import unittest

import torch

from data.utils import load_dataset
from src.preprocess import NIDS_Preprocessor
from src.torch_models.autoencoder import AutoEncoder

class TestTraining(unittest.TestCase):

    def setUp(self) -> None:
        
        X, y = load_dataset("cicids2017", use_subsample=True, subsample_size=0.01)
        X = NIDS_Preprocessor().fit_transform(X).to_numpy()
        # X = X[y == 0]
        self.X = torch.from_numpy(X).float()

    def test_autoencoder(self):

        model = AutoEncoder(num_features=self.X.shape[1])

        model.fit(self.X, num_epochs=3)


