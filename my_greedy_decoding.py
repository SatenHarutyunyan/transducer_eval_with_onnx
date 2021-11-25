from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.common.parts.rnn import label_collate
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, ElementType, HypothesisType, LengthsType, NeuralType
from nemo.utils import logging
import copy

def pack_hypotheses(hyp: rnnt_utils.Hypothesis, logitlen: torch.Tensor,) -> rnnt_utils.Hypothesis:
    if hasattr(logitlen, 'cpu'):
        logitlen_cpu = logitlen.to('cpu')
    else:
        logitlen_cpu = logitlen
    
    hyp.y_sequence = torch.tensor(hyp.y_sequence, dtype=torch.long)
    hyp.length = logitlen_cpu[0]

    if hyp.dec_state is not None:
        hyp.dec_state = _states_to_device(hyp.dec_state)

    return hyp


def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


class _GreedyRNNTInfer(Typing):
    """A greedy transducer decoder.

    Provides a common abstraction for sample level and batch level greedy decoding.

    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        preserve_alignments: Bool flag which preserves the history of alignments generated during
            greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of ints.

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "partial_hypotheses": [NeuralType(elements_type=HypothesisType(), optional=True)],  # must always be last
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
    ):
        super().__init__()
        self.decoder = decoder_model
        self.joint = joint_model

        self._blank_index = blank_index
        self._SOS = blank_index  # Start of single index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        hidden: Optional[torch.Tensor],
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Common prediction step based on the AbstractRNNTDecoder implementation.

        Args:
            label: (int/torch.Tensor): Label or "Start-of-Signal" token.
            hidden: (Optional torch.Tensor): RNN State vector
            add_sos (bool): Whether to add a zero vector at the begging as "start of sentence" token.
            batch_size: Batch size of the output tensor.

        Returns:
            g: (B, U, H) if add_sos is false, else (B, U + 1, H)
            hid: (h, c) where h is the final sequence hidden state and c is
                the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)
        """
        if isinstance(label, torch.Tensor):
            # label: [batch, 1]
            if label.dtype != torch.long:
                label = label.long()

        else:
            # Label is an integer
            if label == self._SOS:
                return self.decoder.predict(None, hidden, add_sos=add_sos, batch_size=batch_size)

            label = label_collate([[label]])

        # output: [B, 1, K]
        return self.decoder.predict(label, hidden, add_sos=add_sos, batch_size=batch_size)

    def _joint_step(self, enc, pred, log_normalize: Optional[bool] = None):
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model. A torch.Tensor of shape [B, 1, H1]
            pred: Output of the Decoder model. A torch.Tensor of shape [B, 1, H2]
            log_normalize: Whether to log normalize or not. None will log normalize only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        with torch.no_grad():
            logits = self.joint.joint(enc, pred)

            if log_normalize is None:
                if not logits.is_cuda:  # Use log softmax only if on CPU
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)
            else:
                if log_normalize:
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits


class GreedyRNNTInfer(_GreedyRNNTInfer):
    """A greedy transducer decoder.

    Sequence level greedy decoding, performed auto-repressively.

    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        preserve_alignments: Bool flag which preserves the history of alignments generated during
            greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of ints.

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.
    """

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
            preserve_alignments=preserve_alignments,
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        previous_kept_hyps_and_maybe_cache: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        encoder_output = torch.tensor(encoder_output)
        with torch.no_grad():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)

            self.decoder.eval()
            self.joint.eval()

            
            # Process each sequence independently
            with self.decoder.as_frozen(), self.joint.as_frozen():
                    inseq = encoder_output[0, :, :].unsqueeze(1)  # [T, 1, D]
                    logitlen = encoded_lengths[0]
                    # partial_hypothesis = partial_hypotheses[0] if partial_hypotheses is not None else None
                    hypothesis, hyp_for_returning = self._greedy_decode(inseq, logitlen, previous_hyp = previous_kept_hyps_and_maybe_cache)

            # Pack results into Hypotheses
            packed_result = pack_hypotheses(hypothesis, encoded_lengths)

        return packed_result, hyp_for_returning

    @torch.no_grad()
    def _greedy_decode(
        self, x: torch.Tensor, out_len: torch.Tensor,
        previous_hyp: Optional[rnnt_utils.Hypothesis] = None
    ):
        # x: [T, 1, D]
        # out_len: [seq_len]
        if previous_hyp is None:
            hypothesis = rnnt_utils.Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestep=[])
        else: 
            hypothesis = previous_hyp

        # For timestep t in X_t
        for time_idx in range(out_len):
            # Extract encoder embedding at timestep t
            # f = x[time_idx, :, :].unsqueeze(0)  # [1, 1, D]
            f = x.narrow(dim=0, start=time_idx, length=1)

            # Setup exit flags and counter
            not_blank = True
            symbols_added = 0

            # While blank is not predicted, or we dont run out of max symbols per timestep
            while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                # In the first timestep, we initialize the network with RNNT Blank
                # In later timesteps, we provide previous predicted label as input.
                last_label = (
                    self._SOS
                    if (hypothesis.y_sequence == [] and hypothesis.dec_state is None)
                    else hypothesis.y_sequence[-1]
                )

                # Perform prediction network and joint network steps.
                g, hidden_prime = self._pred_step(last_label, hypothesis.dec_state)
                logp = self._joint_step(f, g, log_normalize=None)[0, 0, 0, :]

                del g

                # torch.max(0) op doesnt exist for FP 16.
                if logp.dtype != torch.float32:
                    logp = logp.float()

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()  # K is the label at timestep t_s in inner loop, s >= 0.

                if self.preserve_alignments:
                    # insert logits into last timestep
                    hypothesis.alignments[-1].append(k)

                del logp

                # If blank token is predicted, exit inner loop, move onto next timestep t
                if k == self._blank_index:
                    not_blank = False

                    if self.preserve_alignments:
                        # convert Ti-th logits into a torch array
                        hypothesis.alignments.append([])  # blank buffer for next timestep
                else:
                    # Append token to label set, update RNN state.
                    hypothesis.y_sequence.append(k)
                    hypothesis.score += float(v)
                    hypothesis.timestep.append(time_idx)
                    hypothesis.dec_state = hidden_prime

                # Increment token counter.
                    symbols_added += 1
                # print('not_blank',not_blank)
        hyp_for_returning = copy.deepcopy(hypothesis)
        # Remove trailing empty list of Alignments
        if self.preserve_alignments:
            if len(hypothesis.alignments[-1]) == 0:
                del hypothesis.alignments[-1]

        # Unpack the hidden states
        hypothesis.dec_state = self.decoder.batch_select_state(hypothesis.dec_state, 0)

        return hypothesis, hyp_for_returning

