import numpy as np

from src.kitnet_utils.utils.nn import sigmoid, rmse

class KitNetModelParameters:

    def __init__(
            self, 
            num_features: int=5,
            num_hidden: int=3,
            lr: float=1e-3,
            corruption_level: float=0.0,
            training_period: int=10_000,
            hidden_ratio: float | None=None
            ):
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.lr = lr
        self.corruption_level = corruption_level
        self.training_period = training_period
        self.hidden_ratio = hidden_ratio

class KitNetAE:

    def __init__(self, params: KitNetModelParameters, rng_seed: int=1234, epsilon: float=1e-12):
        
        self.params = params
        self.rng_seed = rng_seed
        self.epsilon = epsilon

        # Set local random seed
        self.rng = np.random.RandomState(rng_seed)

        if self.params.hidden_ratio is not None:
            self.params.num_hidden = int(np.ceil(self.params.num_features*self.params.hidden_ratio))

        # Number of processed samples
        self.num_processed = 0

        # Initialize normalization parameters for minmax scaling
        self.norm_max = np.ones((self.params.num_features)) * -np.inf
        self.norm_min = np.ones((self.params.num_features)) * np.inf

        # Initialize parameters uniformly in [-1/F,1/F], with F being the number of features 
        a = 1. / self.params.num_features
        
        # First layer: num_features -> num_hidden
        self.weights_1 = self.rng.uniform(
            low=-a,
            high=a,
            size=(self.params.num_features, self.params.num_hidden)
        )
        self.bias_1 = np.zeros(self.params.num_hidden)
        
        # Second layer: num_hidden -> num_features
        self.weights_2 = self.weights_1.T  # weights_2 always points to the same memory as weights_1 but with transposed view
        self.bias_2 = np.zeros(self.params.num_features)

    def minmax_scale(self, x: np.ndarray):
        return (x - self.norm_min) / (self.norm_max - self.norm_min + self.epsilon)

    def get_perturbed_input(self, x: np.ndarray, corruption_level: float):
        assert corruption_level < 1.

        # Set random input values to zero
        return self.rng.binomial(
            size=x.shape,
            n=1,
            p=1-corruption_level
        ) * x
    
    def get_hidden_values(self, x: np.ndarray):
        return sigmoid(x @ self.weights_1 + self.bias_1)
    
    def get_reconstructed_input(self, hidden: np.ndarray):
        return sigmoid(hidden @ self.weights_2 + self.bias_2)
    
    def train(self, x: np.ndarray):
        self.num_processed += 1

        # Update norms
        self.norm_max[x > self.norm_max] = x[x > self.norm_max]
        self.norm_min[x < self.norm_min] = x[x < self.norm_min]

        # MinMax scaling
        x = self.minmax_scale(x)

        if self.params.corruption_level > 0.0:
            x_perturbed = self.get_perturbed_input(x, self.params.corruption_level)
        else:
            x_perturbed = x

        # Compression and reconstruction
        latent = self.get_hidden_values(x_perturbed)
        x_reconstructed = self.get_reconstructed_input(latent)

        error = x - x_reconstructed
        latent_error = error @ self.weights_1 * (latent - latent**2)
        
        weight_update = np.outer(x_perturbed, latent_error) + np.outer(error.T, latent)
        bias1_update = np.mean(latent_error, axis=0)
        bias2_update = np.mean(error, axis=0)

        self.weights_1 += self.params.lr * weight_update
        # NOTE: since weights_2 points to the same memory, it gets automatically updated
        self.bias_1 += self.params.lr * bias1_update
        self.bias_2 += self.params.lr * bias2_update

        loss = rmse(x, x_reconstructed)
        return loss
    
    def reconstruct(self, x: np.ndarray):
        return self.get_reconstructed_input(self.get_hidden_values(x))
    
    def execute(self, x: np.ndarray):

        if self.num_processed < self.params.training_period:
            return 0.
        
        x = self.minmax_scale(x)
        x_reconstructed = self.reconstruct(x)
        loss = rmse(x, x_reconstructed)
        return loss
        

    def in_training_period(self):
        return self.num_processed < self.params.training_period

