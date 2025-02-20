import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class AutoEncoder(nn.Module):

    def __init__(self, num_features: int, latent_dim: int=16, noise_stddev: float=0.1):
        """
            AutoEncoder model implementation, adapted from Choi H, Unsupervised learning approach for network intrusion detection system using autoencoders. The Journal of Supercomputing. 2019.
            
            The default parameter values for `latent_dim` and `noise_stddev` are taken from the original paper.

            Args:
                num_features (int): Number of input features.
                latent_dim (int): Size of the latent dimensions. Default: 16.
                noise_stddev (float): Standard deviation of the noise introduced in the latent dimension. Default: 0.1.
        """

        super().__init__()

        self.num_features = num_features
        self.latent_dim = latent_dim
        self.noise_stddev = noise_stddev

        self.encoder = nn.Sequential(
            nn.Linear(num_features, latent_dim)
        )

        self.decoder =  nn.Sequential(
            nn.Linear(latent_dim, num_features)
        )

        self.is_fitted = False

    def forward(self, x: torch.Tensor):
        z = self.encoder(x) 
        if self.training and self.noise_stddev > 0:
            z += torch.randn_like(z) * self.noise_stddev
        return self.decoder(z)
    
    def fit(
            self, X, num_epochs: int=10, batch_size: int=256, learning_rate: float=0.0001, 
            validation_split: float=0.2, regularization_weight: float=0.01, 
            verbose: bool=False
        ):
        """
            Trains the autoencoder model.

            Args:
                X (torch.Tensor): Input data.
                num_epochs (int): Number of epochs for training. Default: 10.
                batch_size (int): Batch size for training. Default: 256.
                learning_rate (float): Learning rate for training. Default: 0.0001.
                validation_split (float): Percentage of data to use for validation. Default: 0.2.
                regularization_weight (float): Weight for L2 regularization. Default: 0.01.
                verbose (bool): Whether to print training progress. Default: False.
        """

        X_train, X_val = train_test_split(X, test_size=validation_split, shuffle=False)

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)  

        train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(X_val, batch_size=256, shuffle=False)

        for epoch in range(num_epochs):
            running_loss = 0
            for x_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(x_batch)
                loss = torch.mean((outputs - x_batch)**2)
                if regularization_weight > 0:  # no point in doing extra computation if weight is 0
                    loss += regularization_weight * torch.sum(self.encoder[0].weight**2)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # NOTE: This is not perfectly accurate, since the last batch may have a different size.
            # However, for large datasets, it will be very close to the true loss.
            loss = running_loss / len(train_loader)

            with torch.no_grad():
                val_loss = 0
                for x_batch in val_loader:
                    outputs = self(x_batch)
                    val_loss += torch.mean((outputs - x_batch)**2).item()

            val_loss /= len(val_loader)

            if verbose:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")

        self.is_fitted = True
    
    def evaluate(self, X, batch_size: int=256) -> float:
        """
            Evaluates the autoencoder model.

            Args:
                X (ArrayLike): Input data.
                batch_size (int): Batch size for evaluation. Default: 256.

            Returns:
                float: Reconstruction loss.
        """

        self.eval()
        eval_loader = DataLoader(X, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            loss = 0
            for x_batch in eval_loader:
                outputs = self(x_batch)
                loss += torch.mean((outputs - x_batch)**2).item()
        return loss

    def evaluate_errors(self, X, batch_size: int=256) -> np.ndarray:
        """
            Evaluates the autoencoder model and returns the reconstruction errors.

            Args:
                X (ArrayLike): Input data.
                batch_size (int): Batch size for evaluation. Default: 256.

            Returns:
                np.array: Reconstruction errors.
        """

        self.eval()
        eval_loader = DataLoader(X, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            errors = []
            for x_batch in eval_loader:
                outputs = self(x_batch)
                errors.append(torch.mean((outputs - x_batch)**2, dim=1))

        errors = torch.cat(errors)
        return errors.numpy()
    

class StackedAutoEncoder(AutoEncoder):

    def __init__(self, num_features: int, latent_dim: int=16, noise_stddev: float=0.1):
        """
            AutoEncoder model implementation, adapted from Choi H, Unsupervised learning approach for network intrusion detection system using autoencoders. The Journal of Supercomputing. 2019.
            
            The default parameter values for `latent_dim` and `noise_stddev` are taken from the original paper.

            Args:
                num_features (int): Number of input features.
                latent_dim (int): Size of the latent dimensions. Default: 16.
                noise_stddev (float): Standard deviation of the noise introduced in the latent dimension. Default: 0.1.
        """

        super().__init__()

        self.num_features = num_features
        self.latent_dim = latent_dim
        self.noise_stddev = noise_stddev

        self.encoder = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.decoder =  nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_features)
        )

        self.is_fitted = False