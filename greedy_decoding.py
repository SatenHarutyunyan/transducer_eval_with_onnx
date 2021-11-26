
from typing import List, Optional
import torch
from abc import ABC
from nemo.collections.asr.modules import rnnt_abstract
from rnnt_wer_bpe import Hypothesis
from nemo.collections.common.parts.rnn import label_collate

def pack_hypotheses(hyp: Hypothesis, logitlen: torch.Tensor,) -> Hypothesis:
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


class _GreedyRNNTInfer(ABC):
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
        previous_kept_hyps_and_maybe_cache: Optional[List[Hypothesis]] = None,
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

  