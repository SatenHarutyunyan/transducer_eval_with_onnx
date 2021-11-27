
from typing import  Optional, Tuple
import numpy as np
import torch


def rnn(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    norm: Optional[str] = None,
    forget_gate_bias: Optional[float] = 1.0,
    dropout: Optional[float] = 0.0,
    norm_first_rnn: Optional[bool] = None,
    t_max: Optional[int] = None,
    weights_init_scale: float = 1.0,
    hidden_hidden_bias_scale: float = 0.0,
) -> torch.nn.Module:
    """
    Utility function to provide unified interface to common LSTM RNN modules.

    Args:
        input_size: Input dimension.

        hidden_size: Hidden dimension of the RNN.

        num_layers: Number of RNN layers.

        norm: Optional string representing type of normalization to apply to the RNN.
            Supported values are None, batch and layer.

        forget_gate_bias: float, set by default to 1.0, which constructs a forget gate
                initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

        dropout: Optional dropout to apply to end of multi-layered RNN.

        norm_first_rnn: Whether to normalize the first RNN layer.

        t_max: int value, set to None by default. If an int is specified, performs Chrono Initialization
            of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
            of training.
            Reference:
            [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)

        weights_init_scale: Float scale of the weights after initialization. Setting to lower than one
            sometimes helps reduce variance between runs.

        hidden_hidden_bias_scale: Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
            the default behaviour.

    Returns:
        A RNN module
    """
    if norm not in [None, "batch", "layer"]:
        raise ValueError(f"unknown norm={norm}")

    if norm is None:
        return LSTMDropout(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
            weights_init_scale=weights_init_scale,
            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
        )

    if norm == "batch":
        return BNRNNSum(
            input_size=input_size,
            hidden_size=hidden_size,
            rnn_layers=num_layers,
            batch_norm=True,
            dropout=dropout,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
            norm_first_rnn=norm_first_rnn,
            weights_init_scale=weights_init_scale,
            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
        )

    if norm == "layer":
        return torch.jit.script(
            ln_lstm(  # torch.jit.script(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                forget_gate_bias=forget_gate_bias,
                t_max=t_max,
                weights_init_scale=weights_init_scale,
                hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            )
        )


class LSTMDropout(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: Optional[float],
        forget_gate_bias: Optional[float],
        t_max: Optional[int] = None,
        weights_init_scale: float = 1.0,
        hidden_hidden_bias_scale: float = 0.0,
    ):
        """Returns an LSTM with forget gate bias init to `forget_gate_bias`.
        Args:
            input_size: See `torch.nn.LSTM`.
            hidden_size: See `torch.nn.LSTM`.
            num_layers: See `torch.nn.LSTM`.
            dropout: See `torch.nn.LSTM`.

            forget_gate_bias: float, set by default to 1.0, which constructs a forget gate
                initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

            t_max: int value, set to None by default. If an int is specified, performs Chrono Initialization
                of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
                of training.
                Reference:
                [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)

            weights_init_scale: Float scale of the weights after initialization. Setting to lower than one
                sometimes helps reduce variance between runs.

            hidden_hidden_bias_scale: Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
                the default behaviour.

        Returns:
            A `torch.nn.LSTM`.
        """
        super(LSTMDropout, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
        )

        if t_max is not None:
            # apply chrono init
            for name, v in self.lstm.named_parameters():
                if 'bias' in name:
                    p = getattr(self.lstm, name)
                    n = p.nelement()
                    hidden_size = n // 4
                    p.data.fill_(0)
                    p.data[hidden_size : 2 * hidden_size] = torch.log(
                        torch.nn.init.uniform_(p.data[0:hidden_size], 1, t_max - 1)
                    )
                    # forget gate biases = log(uniform(1, Tmax-1))
                    p.data[0:hidden_size] = -p.data[hidden_size : 2 * hidden_size]
                    # input gate biases = -(forget gate biases)

        elif forget_gate_bias is not None:
            for name, v in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size] *= float(hidden_hidden_bias_scale)

        self.dropout = torch.nn.Dropout(dropout) if dropout else None

        for name, v in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                v.data *= float(weights_init_scale)

    def forward(
        self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, h = self.lstm(x, h)

        if self.dropout:
            x = self.dropout(x)

        return x, h


def label_collate(labels, device=None):
    """Collates the label inputs for the rnn-t prediction network.
    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.
        device: Optional torch device to place the label on.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")

    batch_size = len(labels)
    max_len = max(len(label) for label in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    labels = torch.tensor(cat_labels, dtype=torch.int64, device=device)

    return labels
