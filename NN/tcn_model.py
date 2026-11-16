"""Causal TCN for ego-centric lateral path planning.

Expected input shape: [batch, time, features]
Default input:        [batch, 40, 16]
Default output:       [batch, 4]

The output contract is fixed to the 2.5 s terminal state:
    [y_T, y_dot_T, y_ddot_T, log_sigma_y]
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """Conv1d with left-only padding, so no future sample is used."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias,
        )
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name!r}. Use 'relu' or 'silu'.")


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels independently at every timestep.

    nn.GroupNorm(1, channels) is not suitable here because it computes its
    statistics over both channels and time. Consequently, later samples in an
    input window can change features at earlier timesteps. This layer keeps
    normalization causal by normalizing only the channel vector at each time.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        # [batch, channels, time] -> [batch, time, channels]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class TemporalResidualBlock(nn.Module):
    """Residual TCN block containing two causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.norm1 = ChannelLayerNorm(out_channels)
        self.act1 = make_activation(activation)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation
        )
        self.norm2 = ChannelLayerNorm(out_channels)
        self.act2 = make_activation(activation)
        self.drop2 = nn.Dropout(dropout)

        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.out_activation = make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)
        y = self.drop2(y)

        return self.out_activation(y + residual)


class LateralPlanningTCN(nn.Module):
    """TCN that predicts local lateral-planning parameters."""

    def __init__(
        self,
        input_features: int = 16,
        output_features: int = 4,
        channels: Sequence[int] = (32, 48, 64, 64),
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 3,
        head_hidden: Sequence[int] = (128, 64, 32),
        dropout: float = 0.10,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if len(channels) != len(dilations):
            raise ValueError("channels and dilations must have equal lengths")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")

        blocks: list[nn.Module] = []
        in_channels = input_features
        for out_channels, dilation in zip(channels, dilations):
            blocks.append(
                TemporalResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                )
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)

        head_layers: list[nn.Module] = []
        head_in = channels[-1]
        for head_out in head_hidden:
            head_layers.extend(
                [
                    nn.Linear(head_in, head_out),
                    make_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            head_in = head_out
        head_layers.append(nn.Linear(head_in, output_features))
        self.head = nn.Sequential(*head_layers)

        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.convolutions_per_block = 2

    @property
    def receptive_field(self) -> int:
        return 1 + self.convolutions_per_block * (self.kernel_size - 1) * sum(
            self.dilations
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected [batch, time, features], received {tuple(x.shape)}"
            )
        if x.shape[-1] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} features, received {x.shape[-1]}"
            )

        # Conv1d expects [batch, channels, time].
        x = x.transpose(1, 2)
        features = self.tcn(x)

        # Causal prediction at the current (last) timestep.
        current_features = features[:, :, -1]
        return self.head(current_features)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(7)

    model = LateralPlanningTCN()
    example = torch.randn(8, 40, 16)
    output = model(example)

    print(model)
    print(f"Input shape:          {tuple(example.shape)}")
    print(f"Output shape:         {tuple(output.shape)}")
    print(f"Receptive field:      {model.receptive_field} timesteps")
    print(f"History represented:  {40 / 20:.2f} s at 20 Hz")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

