"""Wrappers for using NNs with scikit-learn API (based on skorch).

TODO
 - checkpoint best epoch iteration based on train_loss (with a callback?);
 - use the best iteration for predictions;

"""
import random
from typing import Callable

import skorch
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hpt.utils import import_object


class _FFNN(nn.Module):
    """Generic feed forward neural network using pytorch.
    https://github.com/AndreFCruz/bias-detection/blob/master/src/architectures/ffnn_architectures.py
    """

    DEFAULT_INPUT_DIM = 1   # Must be a valid value

    def __init__(
            self,
            input_dim: int = DEFAULT_INPUT_DIM,
            output_dim: int = 1,
            hidden_layers: list[int] = None,
            dropout: float = None,
            activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
            use_batch_norm: bool = False,
        ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = []
        assert input_dim > 0 and output_dim > 0

        self.activ_function = activation
        self.output_activ_function = torch.sigmoid if output_dim == 1 else F.softmax

        self.linear_layers, self.bn_layers = self.make_linear_layers(
            input_dim, output_dim, hidden_layers, use_batch_norm,
        )

        if dropout is None:
            # No dropout
            self.dropout_layers = None
        else:
            # Same dropout on all layers
            self.dropout_layers = nn.ModuleList(
                [nn.Dropout(p=dropout) for _ in range(len(hidden_layers))]
            )

        assert (len(self.linear_layers) - 1) == len(hidden_layers)
        assert self.bn_layers is None or len(self.bn_layers) == len(hidden_layers)
        assert self.dropout_layers is None or len(self.dropout_layers) == len(hidden_layers)

    @staticmethod
    def make_linear_layers(
            input_size: int,
            output_size: int,
            hidden_layers: list[int],
            use_batch_norm: bool = False,
        ) -> tuple[nn.ModuleList, nn.ModuleList]:
        """Dynamically construct linear layers.

        Parameters
        ----------
        input_size : int
            The (single-dimensional) shape of the input data.
            The number of neurons in the input layer.
        output_size : int
            The number of neurons in the output layer.
        hidden_layers : List[int]
            The number of neurons per hidden layer, ordered.
            E.g., [20, 10, 5] will generate three hidden layers, respectively
            with 20 -> 10 -> 5 neurons, plus the input and output layers.
        use_batch_norm : bool
            Whether to use batch normalization after all layers.

        Returns
        -------
        A Module corresponding to the constructed Linear layers, as well as a
        second module corresponding to the BatchNorm layers (if applicable).
        """
        linear_layers = []
        bn_layers = []

        # 1st layer maps the input dimension to the 1st hidden/intermediate layer
        neurons_per_layer = [input_size] + hidden_layers

        for idx, in_neurons in enumerate(neurons_per_layer):

            # If this is not the last layer:
            if idx < len(neurons_per_layer) - 1:
                out_neurons = neurons_per_layer[idx+1]

                linear_layers.append(
                    nn.Linear(
                        in_neurons,
                        out_neurons,
                        bias=False if use_batch_norm else True,
                        # batch norm eliminates the need for a bias vector
                        # (it already has a bias vector to shift the weights)
                    ))

                if use_batch_norm:
                    bn_layers.append(nn.BatchNorm1d(out_neurons))
            
            # Else (this is the output layer):
            else:
                # Output layer doesn't usually use batch norm
                linear_layers.append(
                    nn.Linear(in_neurons, output_size, bias=True)
                )

        # NOTE: needs to be wrapped on a ModuleList so the parameters can be found by the optimizer
        return (
            nn.ModuleList(linear_layers),
            nn.ModuleList(bn_layers) if use_batch_norm else None,
        )

    def forward(self, data, sample_weight=None):
        # NOTE:
        # > ignoring `sample_weight` here, will be used when computing the loss
        # > i.e., at `FeedForwardClassifier.get_loss(...)`

        for idx, lin_layer in enumerate(self.linear_layers):
            data = lin_layer(data)
            if idx == len(self.linear_layers) - 1:
                break

            if self.bn_layers is not None and idx < len(self.bn_layers):
                data = self.bn_layers[idx](data)

            data = self.activ_function(data)
            if self.dropout_layers is not None and idx < len(self.dropout_layers):
                data = self.dropout_layers[idx](data)

        return self.output_activ_function(data)


class FeedForwardClassifier(skorch.NeuralNetBinaryClassifier):
    """Skorch wrapper for a standard feed forward neural network.

    Notes
    -----
    Use `module__[kwarg]=[value]` to pass key-word arguments to the torch module.
    This is also available for `criterion__[kwarg]` and `optimizer__[kwarg]`.
    See: https://skorch.readthedocs.io/en/stable/user/neuralnet.html?highlight=module__#special-arguments

    In order to use `sample_weight` when computing the NN loss, we needed to do
    several odd changes to this class; more information at:
    https://skorch.readthedocs.io/en/stable/user/FAQ.html?highlight=sample_weight#i-want-to-use-sample-weight-how-can-i-do-this

    Note that the `sample_weight` fit parameter is required for compatibility
    with fairlearn's meta/ensemble algorithms (EG and GS).
    """

    def __init__(
            self,
            criterion: type | str = nn.BCELoss,
            optimizer: type | str = torch.optim.Adam,
            device: str | torch.device = None,
            criterion__reduction="none",
            **kwargs,
        ):

        # Set torch device to use
        if not isinstance(device, torch.device):
            assert device is None or isinstance(device, str)

            if device:                                  # use given device (if specified)
                device = torch.device(device)
            elif torch.cuda.is_available():             # else, use CUDA if present
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():     # else, use MPS if present
                device = torch.device("mps")
            else:
                # device = torch.device("cpu")          # let skorch handle device=None
                pass

        print(f"Using torch device '{device}'")

        # Load types if passed as a string (this enables choosing from the yaml)
        if isinstance(criterion, str):
            criterion = import_object(criterion)

        if isinstance(optimizer, str):
            optimizer = import_object(optimizer)

        # By default, use no train/valid split
        kwargs.setdefault("train_split", None)

        # Use early stopping if validation data is available
        callbacks = [
            skorch.callbacks.ProgressBar(),
            skorch.callbacks.InputShapeSetter(
                param_name="input_dim",
                input_dim_fn=FeedForwardClassifier._input_dim_fn,
            )
        ]
        if kwargs["train_split"] is not None:
            callbacks += [skorch.callbacks.EarlyStopping()]

        # Set important defaults
        kwargs.setdefault("module", _FFNN)
        kwargs.setdefault("device", device)
        kwargs.setdefault("criterion", criterion)
        kwargs.setdefault("optimizer", optimizer)
        kwargs.setdefault("callbacks", callbacks)
        kwargs.setdefault("criterion__reduction", criterion__reduction)

        # NOTE
        # - The reason why these kwargs are provided as defaults instead of 
        # directly in the `__init__` is because some meta algorithms (e.g., 
        # fairlearn EG and GS) provide all model parameters and attributes as 
        # kwargs when trying to replicate a model object;
        # - Essentially, for example the parameter `module` would show up in the
        # kwargs dict, even though it's not part of this method's init (as the
        # module should always be the same `_FFNN`);
        # - It's weird but it works I guess :)

        super().__init__(**kwargs)

    @staticmethod
    def _input_dim_fn(X: pd.DataFrame | np.ndarray | dict) -> int:
        """Returns the input dimension of the provided data.

        Notes
        -----
        Needed to customize this function to be compatible with dictionary
        inputs (which was needed for compatibility with `sample_weight`).

        Based on:
        https://github.com/skorch-dev/skorch/blob/fc3758bf48ccd9b6b1901694a9f2520066a476ce/skorch/callbacks/training.py#L846
        """
        if isinstance(X, dict):
            return FeedForwardClassifier._input_dim_fn(X["data"])

        if len(X.shape) < 2:
            raise ValueError(
                "Expected at least two-dimensional input data for X. "
                "If your data is one-dimensional, please use the "
                "`input_dim_fn` parameter to infer the correct "
                "input shape."
            )

        return X.shape[-1]

    @staticmethod
    def seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def convert_input_data(data) -> np.ndarray:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.to_numpy(dtype=np.float32)
        elif isinstance(data, np.ndarray):
            return data.astype(np.float32)
        else:
            # raise ValueError(f"Invalid input data type: {type(data)}.")
            return data

    def initialize(self):
        # self.seed(self.random_state)
        super().initialize()

    def fit(self, X, y, sample_weight=None, **fit_params):
        X, y = map(self.convert_input_data, [X, y])

        if sample_weight is not None:
            X = {
                "data": X,
                "sample_weight": sample_weight,
            }

            # TODO: check these prints in the cluster logs to check if the
            # `sample_weight`` indeed changes when running EG or GS
            print("random sample_weight:", sample_weight.sample(n=1).iloc[0])

        return super().fit(X, y, **fit_params)

    def predict_proba(self, X):
        X = self.convert_input_data(X)
        return super().predict_proba(X)

    def get_loss(self, y_pred, y_true, X: dict, *args, **kwargs):
        # TODO: check if fairlearn EG/GS correctly uses the sample_weight parameter!

        # override get_loss to use the sample_weight from X
        loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
        if isinstance(X, dict) and "sample_weight" in X:
            sample_weight = skorch.utils.to_tensor(X['sample_weight'], device=self.device)
            loss_reduced = (sample_weight * loss_unreduced).mean()
        else:
            loss_reduced = loss_unreduced.mean()

        return loss_reduced
