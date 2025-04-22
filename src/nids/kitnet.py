import logging

# External packages
import numpy as np

# Local modules
from src.kitnet_utils.autoencoder import KitNetAE, KitNetModelParameters
from src.kitnet_utils.correlation_cluster import CorrelationCluster


class KitNET:

    def __init__(
            self, 
            num_features: int, 
            max_autoencoder_size: int=10,
            learning_rate: float=0.1,
            hidden_ratio: float=0.75,
            random_state: int | None=None,
        ):
        """
        Args:
            num_features (int): The number of features for an input sample.
            max_autoencoder_size (int): Maximum number of input features for an autoencoder in the ensemble layer. Used while building the feature map via hierarchical correlation clustering.
            learning_rate (float): Learning rate for SGD.
            hidden_ratio (float): Ratio between features and latent dimension in the autoencoders.
            random_state (float): PRNG seed. If None, a random one is used, leading to non-reproducible behavior.
        """

        self.num_features = num_features

        if max_autoencoder_size < 1:
            self.max_autoencoder_size = 1
        else:
            self.max_autoencoder_size = max_autoencoder_size

        self.learning_rate = learning_rate
        self.hidden_ratio = hidden_ratio


        self.ensemble_layer: list[KitNetAE] = []
        self.output_layer = None

        self.correlation_cluster = CorrelationCluster(self.num_features)

        self.random_state = random_state
            

    def fit(self, X_fm: np.ndarray, X_train: np.ndarray):
        """
        Fits KitNET on a dataset X_fm for learning the feature map and on another dataset X_train for training the autoencoder ensemble.

        Args:
            X_fm (np.ndarray): Array with shape (num_fm, num_features).
            X_train (np.ndarray): Array with shape (num_train, num_features).
        """

        for x in X_fm:
            self.update_feature_map(x)

        self.build_feature_map()

        for x in X_train:
            self.train(x)
    
    def evaluate_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluates the reconstruction errors of KitNET on X.

        Args:
            X (np.ndarray): Array with shape (num_samples, num_features).

        Returns:
            errors (np.ndarray): Array with shape (num_samples,)
        """

        errors = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            errors[i] = self.execute(x)

        return errors

    def update_feature_map(self, x: np.ndarray):
        """
        Updates correlation cluster for feature map learning based on a single sample x.

        Args:
            x (np.ndarray): Array with shape (num_features,).
        """
        self.correlation_cluster.update(x)

    def build_feature_map(self):
        """
        Builds feature map and corresponding autoencoder ensemble.
        """
        self.feature_map = self.correlation_cluster.get_clusters(self.max_autoencoder_size)
        self.build_autoencoder_ensemble()

    def train(self, x: np.ndarray):
        """
        Trains KitNET's autoencoder ensemble on a single sample x.

        Args:
            x (np.ndarray): Array with shape (num_features,).
        """
        errors = np.zeros(len(self.ensemble_layer))
        for i, ae_model in enumerate(self.ensemble_layer):
            x_cluster = x[self.feature_map[i]]
            errors[i] = ae_model.train(x_cluster)

        self.output_layer.train(errors)
    

    def execute(self, x: np.ndarray):
        """
        Predicts the anomaly score of a single sample x.

        Args:
            x (np.ndarray): Array with shape (num_features,).

        Returns:
            rmse (float): RMSE between x and KitNET's reconstruction of x.
        """
        if self.feature_map is None:
            raise RuntimeError('KitNET Cannot execute x, because a feature mapping has not yet been learned or provided. Try running process(x) instead.')

        errors = np.zeros(len(self.ensemble_layer))
        for i, ae_model in enumerate(self.ensemble_layer):
            x_cluster = x[self.feature_map[i]]
            errors[i] = ae_model.execute(x_cluster)

        rmse = self.output_layer.execute(errors)

        return rmse



    def build_autoencoder_ensemble(self):
        """
        Constructs the ensemble layer and output layer of KitNET.
        """
        self.ensemble_layer = []
        for cluster in self.feature_map:
            params = KitNetModelParameters(
                num_features=len(cluster),
                num_hidden=0,
                lr=self.learning_rate,
                corruption_level=0,
                training_period=0,
                hidden_ratio=self.hidden_ratio
            )
            self.ensemble_layer.append(KitNetAE(params, rng_seed=self.random_state))
        
        params = KitNetModelParameters(
            num_features=len(self.ensemble_layer),
            num_hidden=0,
            lr=self.learning_rate,
            corruption_level=0,
            training_period=0,
            hidden_ratio=self.hidden_ratio
        )
        self.output_layer = KitNetAE(params, rng_seed=self.random_state)