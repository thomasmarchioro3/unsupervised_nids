import numpy as np

# Local imports
from src.torch_models.autoencoder import AutoEncoder


class NIDS_AutoEncoder(AutoEncoder):

    def __init__(self, n_features: int, noise_stddev: float=0.0, calibration_strategy: str='z_score', z_threshold: float=3.0):
        """
            NIDS_AutoEncoder class. Extends the AutoEncoder class to include calibration process.

            Possible calibration strategies:
                - z_score: Original calibration strategy from [1]. The threshold is computed as `MEAN(errors) + z_threshold * STD(errors)`.

            [1] Choi et al., Unsupervised learning approach for network intrusion detection system using autoencoders. The Journal of Supercomputing. 2019.
            
            
            Args:
                n_features (int): Number of input features.
                noise_stddev (float): Standard deviation of the noise added to the latent dimension. Default: 0.0.
                calibration_strategy (str): Calibration strategy for the autoencoder. Default: 'z_score'.
                z_score_threshold (float): Z-score threshold for the calibration strategy. Default: 3.0.
                
        """
        super().__init__(n_features)
        self.calibration_strategy = calibration_strategy
        self.noise_stddev = noise_stddev
        self.z_threshold = z_threshold

        self.threshold = None

        self.is_fitted = False
        self.is_calibrated = False

    
    def calibrate(self, X):
        """
            Calibrate the autoencoder model.
            
            Args:
                X (ArrayLike): Input data.

            Raises:
                Exception: If the model is not fitted.
                ValueError: If the calibration strategy is unknown.
        """

        if not self.is_fitted:
            raise Exception('Model not fitted')
        

        match self.calibration_strategy:
            case 'z_score':
                errors = self.evaluate_errors(X)
                # ignore outliers
                errors = errors[errors < np.percentile(errors, 99)]
                self.threshold = np.mean(errors) + self.z_threshold * np.std(errors)

            case _:
                raise ValueError(f'Unknown calibration strategy: {self.calibration_strategy}')
            
        self.is_calibrated = True


    def predict(self, X):
        """
            Predict anomalies in the input data.

            Args:
                X (ArrayLike): Input data.

            Returns:
                np.ndarray: Predicted anomalies (1 for anomaly, 0 for normal).
        """

        assert self.is_fitted, 'Model not fitted'
        assert self.is_calibrated, 'Model not calibrated'

        errors = self.evaluate_errors(X)
        return 1 * (errors > self.threshold)