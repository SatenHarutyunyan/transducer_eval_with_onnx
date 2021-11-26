

from typing import List
from abc import ABC, abstractmethod
from mixins import TokenizerSpec
import editdistance
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = ['AbstractRNNTDecoding','RNNTBPEDecoding','word_error_rate', 'Hypothesis']



@dataclass
class Hypothesis:
    score: float
    y_sequence: Union[List[int], torch.Tensor]
    text: Optional[str] = None
    dec_out: Optional[List[torch.Tensor]] = None
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    timestep: Union[List[int], torch.Tensor] = field(default_factory=list)
    alignments: Optional[Union[List[int], List[List[int]]]] = None
    length: Union[int, torch.Tensor] = 0
    y: List[torch.tensor] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    lm_scores: Optional[torch.Tensor] = None
    tokens: Optional[Union[List[int], torch.Tensor]] = None
    last_token: Optional[torch.Tensor] = None

class AbstractRNNTDecoding(ABC):
    

    def __init__(self, decoding_cfg, decoder, joint, blank_id: int):
        super(AbstractRNNTDecoding, self).__init__()
        self.cfg = decoding_cfg
        self.blank_id = blank_id
        self.compute_hypothesis_token_set = self.cfg.get("compute_hypothesis_token_set", False)

       
    def decode_hypothesis(self, hypotheses_list: List[Hypothesis]) -> List[Hypothesis]:
        """
        Decode a list of hypotheses into a list of strings.

        Args:
            hypotheses_list: List of Hypothesis.

        Returns:
            A list of strings.
        """
        for ind in range(len(hypotheses_list)):
            # Extract the integer encoded hypothesis
            prediction = hypotheses_list[ind].y_sequence

            if type(prediction) != list:
                prediction = prediction.tolist()

            # RNN-T sample level is already preprocessed by implicit CTC decoding
            # Simply remove any blank tokens
            prediction = [p for p in prediction if p != self.blank_id]

            # De-tokenize the integer tokens
            hypothesis = self.decode_tokens_to_str(prediction)
            hypotheses_list[ind].text = hypothesis

            if self.compute_hypothesis_token_set:
                hypotheses_list[ind].tokens = self.decode_ids_to_tokens(prediction)
        return hypotheses_list

    @abstractmethod
    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decoder a token id list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        raise NotImplementedError()

    @abstractmethod
    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Implemented by subclass in order to decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        raise NotImplementedError()

class RNNTBPEDecoding(AbstractRNNTDecoding):
    """
    
    """

    def __init__(self, decoding_cfg, decoder, joint, tokenizer: TokenizerSpec):
        blank_id = tokenizer.tokenizer.vocab_size
        self.tokenizer = tokenizer

        super(RNNTBPEDecoding, self).__init__(
            decoding_cfg=decoding_cfg, decoder=decoder, joint=joint, blank_id=blank_id
        )

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decoder a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis = self.tokenizer.ids_to_text(tokens)
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Implemented by subclass in order to decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        token_list = self.tokenizer.ids_to_tokens(tokens)
        return token_list

def word_error_rate(hypotheses: List[str], references: List[str], use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same
    length.
    Args:
      hypotheses: list of hypotheses
      references: list of references
      use_cer: bool, set True to enable cer
    Returns:
      (float) average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer




