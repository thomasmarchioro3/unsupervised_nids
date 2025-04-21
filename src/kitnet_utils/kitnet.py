import logging

# External packages
import numpy as np

# Local modules
from src.kitnet_utils.autoencoder import KitNetAE, KitNetModelParameters
from src.kitnet_utils.correlation_cluster import CorrelationCluster

logger = logging.Logger(name="kitnet")

class KitNET:

    def __init__(
            self, 
            num_features: int, 
            max_autoencoder_size: int=10,
            training_period: int=10_000,
            feature_map_learning_period: int | None=None,
            learning_rate: float=0.1,
            hidden_ratio: float=0.75,
            feature_map: list | None=None
        ):
        """
        Args:
            num_features (int): The number of features for an input sample.
            max_autoencoder_size (int): Maximum number of input features for an autoencoder in the ensemble layer. Used while building the feature map via hierarchical correlation clustering.
            training_period (int): Number of samples used for training before switching to inference mode.
            feature_map_learning_period (int, optional): Number of samples used for learning the feature map. If None, it is set equal to the training_period.
            learning_rate (float): Learning rate for SGD.
            hidden_ratio (float): Ratio between features and latent dimension in the autoencoders.
            feature_map (list of list of int, optional): The feature map can be directly specified instead of being learned. Example: [[2,5,3],[4,0,1],[6,7]]. 
        """

        logger.addHandler(logging.StreamHandler())

        self.num_features = num_features

        if max_autoencoder_size < 1:
            self.max_autoencoder_size = 1
        else:
            self.max_autoencoder_size = max_autoencoder_size

        self.training_period = training_period
        if feature_map_learning_period is None:
            self.feature_map_learning_period = training_period
        else:
            self.feature_map_learning_period = feature_map_learning_period

        self.learning_rate = learning_rate
        self.hidden_ratio = hidden_ratio

        self.num_train = 0
        self.num_inferences = 0

        self.feature_map = feature_map

        self.ensemble_layer: list[KitNetAE] = []
        self.output_layer = None

        if feature_map is not None:
            logger.log(level=logging.INFO, msg="Feature-Mapper: inference-mode, Anomaly-Detector: train-mode")
            self.feature_map_learning_period = 0
            self.build_autoencoder_ensemble()
        else:
            logger.log(level=logging.INFO, msg="Feature-Mapper: train-mode, Anomaly-Detector: off-mode")
            self.correlation_cluster = CorrelationCluster(self.num_features)
            

    def process(self, x: np.ndarray):
        """
        Processes (trains of predicts anomaly score) a single sample x.

        Args:
            x (np.ndarray): Array with shape (num_features,).
        """
        if self.num_train > self.feature_map_learning_period + self.training_period:
            return self.execute(x)
        
        self.train(x)
        return 0.

    def train(self, x: np.ndarray):
        """
        Trains KitNet on a single sample x.

        Args:
            x (np.ndarray): Array with shape (num_features,).
        """

        if self.num_train <= self.feature_map_learning_period and self.feature_map is None:
            # Update correlation cluster
            self.correlation_cluster.update(x)
            if self.num_train == self.feature_map_learning_period:
                # Once the feature map learning period is completed, build the feature map 
                self.feature_map = self.correlation_cluster.get_clusters(self.max_autoencoder_size)
                self.build_autoencoder_ensemble()
                logger.log(
                    level=logging.INFO,
                    msg=f"Feature mapping complete. {self.num_features:d} features mapped to {len(self.feature_map):d} autoencoders."
                )
                logger.log(level=logging.INFO, msg="Feature-Mapper: inference-mode, Anomaly-Detector: train-mode")
        else:
            # Train ensemble of autoencoders
            errors = np.zeros(len(self.ensemble_layer))
            for i, ae_model in enumerate(self.ensemble_layer):
                x_cluster = x[self.feature_map[i]]
                errors[i] = ae_model.train(x_cluster)

            self.output_layer.train(errors)
            if self.num_train == self.training_period + self.feature_map_learning_period:
                logger.log(level=logging.INFO, msg="Feature-Mapper: inference-mode, Anomaly-Detector: inference-mode")
        self.num_train += 1
    

    def execute(self, x: np.ndarray):
        """
        Predicts the anomaly score of a single sample x.

        Args:
            x (np.ndarray): Array with shape (num_features,).
        """
        if self.feature_map is None:
            raise RuntimeError('KitNET Cannot execute x, because a feature mapping has not yet been learned or provided. Try running process(x) instead.')

        self.num_inferences += 1
        errors = np.zeros(len(self.ensemble_layer))
        for i, ae_model in enumerate(self.ensemble_layer):
            x_cluster = x[self.feature_map[i]]
            errors[i] = ae_model.execute(x_cluster)

        return self.output_layer.execute(errors)



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
            self.ensemble_layer.append(KitNetAE(params))
        
        params = KitNetModelParameters(
            num_features=len(self.ensemble_layer),
            num_hidden=0,
            lr=self.learning_rate,
            corruption_level=0,
            training_period=0,
            hidden_ratio=self.hidden_ratio
        )
        self.output_layer = KitNetAE(params)