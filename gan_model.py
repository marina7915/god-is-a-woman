import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm
from .utils import Definitions

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, layer_sizes: list[int], activation: str, batchnorm: bool, dropout_p: float, apply_batchnorm_first: bool = True) -> None:
        super().__init__()
        assert len(layer_sizes) >= 1

        # Encoder
        encoder = []
        if batchnorm and apply_batchnorm_first:
            encoder.append(nn.BatchNorm1d(in_features))
        if dropout_p:
            encoder.append(nn.Dropout(p=dropout_p))
        encoder.append(nn.Linear(in_features, layer_sizes[0]))
        encoder = nn.Sequential(*encoder)
        # Decoder
        decoder = nn.Linear(layer_sizes[-1], out_features)

        # Activation
        activation = Definitions.activations[activation]

        # Layers
        layers = []
        for i in range(len(layer_sizes)):
            layers.append(activation)
            if batchnorm:
                layers.append(nn.BatchNorm1d(layer_sizes[i]))
            if dropout_p:
                layers.append(nn.Dropout(p=dropout_p))
            if i + 1 < len(layer_sizes):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.net = nn.Sequential(
            encoder,
            *layers,
            decoder,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPwithSkipConnections(nn.Module):
    def __init__(self, in_features: int, out_features: int, layer_sizes: list[int], activation: str, batchnorm: bool, dropout_p: float, apply_batchnorm_first: bool = True) -> None:
        super().__init__()
        assert len(layer_sizes) >= 1

        # Encoder
        encoder = []
        if batchnorm and apply_batchnorm_first:
            encoder.append(nn.BatchNorm1d(in_features))
        if dropout_p:
            encoder.append(nn.Dropout(p=dropout_p))
        encoder.append(nn.Linear(in_features, layer_sizes[0]))
        self.encoder = nn.Sequential(*encoder)
        # Decoder
        self.decoder = nn.Linear(layer_sizes[-1] + layer_sizes[0], out_features)

        # Activation
        self.activation = Definitions.activations[activation]

        # Layers
        layers = []
        for i in range(len(layer_sizes)):
            layers.append(self.activation)
            if batchnorm:
                layers.append(nn.BatchNorm1d(layer_sizes[i]))
            if dropout_p:
                layers.append(nn.Dropout(p=dropout_p))
            if i + 1 < len(layer_sizes):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        x = self.layers(encoded)
        x = torch.cat((self.activation(x), encoded), dim=1)
        return self.decoder(x)

class DiscriminatorMLP(nn.Module):
    """
    A discriminator class for a GAN that distinguishes between real and generated data.
    Uses a configurable MLP architecture with optional batch normalization and skip connections.
    """

    def __init__(self, in_features: int, in_generated: int, layer_sizes: list[int], activation: str, batchnorm: bool, dropout_p: float, use_skip_connections: bool) -> None:
        super().__init__()
        # Set up input features and batch normalization
        self.in_features = in_features
        self.in_generated = in_generated
        self.batchnorm = batchnorm
        self.batchnorm_module = nn.BatchNorm1d(in_features)

        # Choose MLP architecture based on whether to use skip connections
        mlp_class = MLPwithSkipConnections if use_skip_connections else MLP
        self.mlp = mlp_class(in_features=in_features + in_generated, out_features=1, layer_sizes=layer_sizes, activation=activation, batchnorm=batchnorm, dropout_p=dropout_p, apply_batchnorm_first=False)

        # Output layer to convert logits to probabilities
        self.to_prob = nn.Sigmoid()

    def forward(self, features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Optionally apply batch normalization to real data features before processing
        if self.batchnorm:
            features = self.batchnorm_module(features)

        # Concatenate real and generated data features for discrimination
        x = torch.cat((features, x), dim=1)

        # Ensure correct input size and apply MLP
        assert x.shape[1] == self.in_features + self.in_generated
        return self.to_prob(self.mlp(x))

class GeneratorMLP(nn.Module):
    """
    A generator class for a GAN that synthesizes realistic data samples from noise input.
    It uses a configurable MLP architecture with optional batch normalization, skip connections, and an additional vector addition.
    Outputs are partially processed with a softplus activation to ensure non-negativity for a subset of features.
    """

    def __init__(self, noise_dim: int, in_features: int, out_features: int, layer_sizes: list[int], activation: str, batchnorm: bool, dropout_p: float, nonnegative_end_ind: int, use_skip_connections: bool, add_vector: list[int] | None = None) -> None:
        super().__init__()
        # Initialize dimensions
        self.noise_dim = noise_dim
        self.in_features = in_features

        # Choose MLP class based on skip connections
        mlp_class = MLPwithSkipConnections if use_skip_connections else MLP
        self.mlp = mlp_class(in_features=in_features + noise_dim, out_features=out_features, layer_sizes=layer_sizes, activation=activation, batchnorm=batchnorm, dropout_p=dropout_p)

        # Ensure that the nonnegative_end_ind is within the valid range
        assert 0 <= nonnegative_end_ind <= out_features
        self.nonnegative_end_ind = nonnegative_end_ind

        # Initialize add_vector if provided
        if add_vector is not None:
            self.register_buffer("add_vector", torch.tensor(add_vector, dtype=torch.float32).reshape(1, -1))
        else:
            self.add_vector = None

    def forward(self, noise: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Check the dimensionality of noise and input features
        assert noise.shape[1] == self.noise_dim
        assert x.shape[1] == self.in_features

        # Concatenate noise and input features
        x = torch.cat((noise, x), dim=1)
        assert x.shape[1] == self.in_features + self.noise_dim

        # Pass through the MLP
        x = self.mlp(x)

        # Apply softplus activation to specified range
        result = F.softplus(x[:, : self.nonnegative_end_ind])
        if self.nonnegative_end_ind != x.shape[1]:
            result = torch.cat((result, x[:, self.nonnegative_end_ind:]), dim=1)

        # Add add_vector if it exists
        if self.add_vector is not None:
            result = result + self.add_vector

        # Return the result
        return result

    def get_noise(self, batch_size) -> torch.Tensor:
        return torch.randn(batch_size, self.noise_dim)

class GeneratorTransformer(nn.Module):
    def __init__(self, noise_dim: int, in_features: int, cond_dim: int, out_features: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, nonnegative_end_ind: int, add_vector: list[int] | None = None) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.in_features = in_features
        self.cond_dim = cond_dim
        self.d_model = d_model
        self.nonnegative_end_ind = nonnegative_end_ind

        self.encoder = nn.Linear(in_features + noise_dim + cond_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        self.decoder = nn.Linear(d_model, out_features)

        if add_vector is not None:
            self.register_buffer("add_vector", torch.tensor(add_vector, dtype=torch.float32).reshape(1, -1))
        else:
            self.add_vector = None

    def forward(self, noise: torch.Tensor, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat((noise, x, cond), dim=1)
        x = self.encoder(x).unsqueeze(1)  # Add sequence dimension
        x = self.pos_encoder(x)
        x = self.transformer(x, x).squeeze(1)  # Transformer expects (S, N, E) shape
        x = self.decoder(x)

        result = F.softplus(x[:, :self.nonnegative_end_ind])
        if self.nonnegative_end_ind != x.shape[1]:
            result = torch.cat((result, x[:, self.nonnegative_end_ind:]), dim=1)

        if self.add_vector is not None:
            result = result + self.add_vector

        return result

    def get_noise(self, batch_size) -> torch.Tensor:
        return torch.randn(batch_size, self.noise_dim)

class DiscriminatorTransformer(nn.Module):
    def __init__(self, in_features: int, in_generated: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.in_generated = in_generated

        self.encoder = spectral_norm(nn.Linear(in_features + in_generated, d_model))
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        self.decoder = spectral_norm(nn.Linear(d_model, 1))

    def forward(self, features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((features, x), dim=1)
        x = self.encoder(x).unsqueeze(1)  # Add sequence dimension
        x = self.pos_encoder(x)
        x = self.transformer(x, x).squeeze(1)  # Transformer expects (S, N, E) shape
        x = self.decoder(x)
        return torch.sigmoid(x)

class GAN(nn.Module):
    def __init__(self, gen_arch: str, disc_arch: str, **kwargs):
        super().__init__()

        if gen_arch == 'MLP':
            self.generator = GeneratorMLP(**kwargs['gen_kwargs'])
        elif gen_arch == 'Transformer':
            self.generator = GeneratorTransformer(**kwargs['gen_kwargs'])
        else:
            raise ValueError("Invalid generator architecture specified")

        if disc_arch == 'MLP':
            self.discriminator = DiscriminatorMLP(**kwargs['disc_kwargs'])
        elif disc_arch == 'Transformer':
            self.discriminator = DiscriminatorTransformer(**kwargs['disc_kwargs'])
        else:
            raise ValueError("Invalid discriminator architecture specified")

    def forward(self, noise: torch.Tensor, real_data: torch.Tensor, cond: torch.Tensor = None):
        fake_data = self.generator(noise, real_data if cond is None else cond)
        disc_real = self.discriminator(real_data, real_data)
        disc_fake = self.discriminator(real_data, fake_data.detach())
        return fake_data, disc_real, disc_fake

    def generate(self, noise: torch.Tensor, cond: torch.Tensor = None):
        return self.generator(noise, cond)