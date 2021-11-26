
import copy
from typing import Any, Dict, List, Optional, Union

import torch
from nemo.collections.asr.modules import rnnt_abstract
from rnnt_wer_bpe import Hypothesis
from abc import ABC

def pack_hypotheses(hyp: Hypothesis) -> Hypothesis:
    hyp.y_sequence = torch.tensor(hyp.y_sequence, dtype=torch.long)

    if hyp.dec_state is not None:
        hyp.dec_state = _states_to_device(hyp.dec_state)

    return hyp


def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


class BeamRNNTInfer(ABC):

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        beam_size: int,
        search_type: str = 'default',
        score_norm: bool = True,
        language_model: Optional[Dict[str, Any]] = None,
        softmax_temperature: float = 1.0
    ):
        self.decoder = decoder_model
        self.joint = joint_model

        self.blank = decoder_model.blank_idx
        self.vocab_size = decoder_model.vocab_size
        self.search_type = search_type
        

        if beam_size < 1:
            raise ValueError("Beam search size cannot be less than 1!")

        self.beam_size = beam_size
        self.score_norm = score_norm

        if self.beam_size == 1:
            logging.info("Beam size of 1 was used, switching to sample level `greedy_search`")
            self.search_algorithm = self.greedy_search
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        else:
            raise NotImplementedError(
                f"The search type ({search_type}) supplied is not supported!\n"
                f"Please use one of : (default, tsd, alsd, nsc)"
            )

        if softmax_temperature != 1.0 and language_model is not None:
            logging.warning(
                "Softmax temperature is not supported with LM decoding." "Setting softmax-temperature value to 1.0."
            )

            self.softmax_temperature = 1.0
        else:
            self.softmax_temperature = softmax_temperature
        self.language_model = language_model
       

   
    def __call__(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        previous_kept_hyps_and_maybe_cache: Optional[List[Hypothesis]] = None,
    ) -> Hypothesis:
        """Perform general beam search.

        Args:
            encoder_output: Encoded speech features (B, T_max, D_enc)
            encoded_lengths: Lengths of the encoder outputs

        Returns:
            Either a list containing a single Hypothesis (when `return_best_hypothesis=True`,
            otherwise a list containing a single NBestHypotheses, which itself contains a list of
            Hypothesis. This list is sorted such that the best hypothesis is the first element.
        """
       
        encoder_output = torch.tensor(encoder_output)
        
        with torch.no_grad():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1,2)  # (B, T, D)
            # print('encoder_output.shape',encoder_output.shape)
          
            self.decoder.eval()
            self.joint.eval()

            # Freeze the decoder and joint to prevent recording of gradients
            # during the beam loop.
            _p = next(self.joint.parameters())
            dtype = _p.dtype
        
            inseq = encoder_output  # [1, T, D]
            logitlen = encoded_lengths
            
            if inseq.dtype != dtype:
                inseq = inseq.to(dtype=dtype)
                

            # Execute the specific search strategy
            nbest_hyps_and_cache = self.search_algorithm(
                inseq, logitlen, previous_kept_hyps_and_maybe_cache = previous_kept_hyps_and_maybe_cache
            )  # sorted list of hypothesis
            # print('type(nbest_hyps_and_cache[0][0])',type(nbest_hyps_and_cache[0][0]))
            # exit('exiting in beam onnx decode')
            best_hypothesis = copy.deepcopy(nbest_hyps_and_cache[0][0])
            
            # Prepare the list of hypotheses
            best_hypothesis = pack_hypotheses(best_hypothesis)
    
        return best_hypothesis, nbest_hyps_and_cache

    def sort_nbest(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Sort hypotheses by score or score given sequence length.

        Args:
            hyps: list of hypotheses

        Return:
            hyps: sorted list of hypotheses
        """
        if self.score_norm:
            return sorted(hyps, key=lambda x: x.score / len(x.y_sequence), reverse=True)
        else:
            return sorted(hyps, key=lambda x: x.score, reverse=True)

    def greedy_search(
        self, h: torch.Tensor, encoded_lengths: torch.Tensor,
        previous_kept_hyps_and_maybe_cache: Optional[Hypothesis] = None
    ) -> List[Hypothesis]:
        """Greedy search implementation for transducer.
        Generic case when beam size = 1. Results might differ slightly due to implementation details
        as compared to `GreedyRNNTInfer` and `GreedyBatchRNNTInfer`.

        Args:
            h: Encoded speech features (1, T_max, D_enc)

        Returns:
            hyp: 1-best decoding results
        """
     
        
        if previous_kept_hyps_and_maybe_cache is None:
             # Initialize zero vector states
            dec_state = self.decoder.initialize_state(h)
            cache = {}
            
            # Initialize first hypothesis for the beam (blank)
            hyp = [Hypothesis(score=0.0, y_sequence=[self.blank], dec_state=dec_state, timestep=[-1], length=0)]
        else:
            hyp = previous_kept_hyps_and_maybe_cache[0]
            cache = previous_kept_hyps_and_maybe_cache[1]

        # Initialize zero state vectors
        dec_state = self.decoder.initialize_state(h)


        # Initialize state and first token
        y, state, _ = self.decoder.score_hypothesis(hyp, cache)

        for i in range(int(encoded_lengths)):
            hi = h[:, i : i + 1, :]  # [1, 1, D]

            not_blank = True
            symbols_added = 0

            while not_blank:
                ytu = torch.log_softmax(self.joint.joint(hi, y) / self.softmax_temperature, dim=-1)  # [1, 1, 1, V + 1]
                ytu = ytu[0, 0, 0, :]  # [V + 1]

                # max() requires float
                if ytu.dtype != torch.float32:
                    ytu = ytu.float()

                logp, pred = torch.max(ytu, dim=-1)  # [1, 1]
                pred = pred.item()


                if pred == self.blank:
                    not_blank = False

            
                else:
                    # Update state and current sequence
                    hyp.y_sequence.append(int(pred))
                    hyp.score += float(logp)
                    hyp.dec_state = state
                    hyp.timestep.append(i)

                    # Compute next state and token
                    y, state, _ = self.decoder.score_hypothesis(hyp, cache)
                symbols_added += 1

    
        return [hyp] ,cache

    def default_beam_search(
        self, h: torch.Tensor, encoded_lengths: torch.Tensor, 
        previous_kept_hyps_and_maybe_cache: Optional[Hypothesis] = None
    ) -> List[Hypothesis]:
        """Beam search implementation.

        Args:
            x: Encoded speech features (1, T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results
        """
        # Initialize states
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))
        blank_tensor = torch.tensor([self.blank], device=h.device, dtype=torch.long)

        # Precompute some constants for blank position
        ids = list(range(self.vocab_size + 1))
        ids.remove(self.blank)

        # Used when blank token is first vs last token
        if self.blank == 0:
            index_incr = 1
        else:
            index_incr = 0

        
        if previous_kept_hyps_and_maybe_cache is None:
             # Initialize zero vector states
            dec_state = self.decoder.initialize_state(h)
            cache = {}
            
            # Initialize first hypothesis for the beam (blank)
            kept_hyps = [Hypothesis(score=0.0, y_sequence=[self.blank], dec_state=dec_state, timestep=[-1], length=0)]
        else:
            kept_hyps = previous_kept_hyps_and_maybe_cache[0]
            cache = previous_kept_hyps_and_maybe_cache[1]
    
        # for hyp in kept_hyps:
        #     hyp.dec_state = tuple([hyp.dec_state[i].to(device='cuda') for i in range(len(hyp.dec_state))])
        for i in range(int(encoded_lengths)):
            hi = h[:, i : i + 1, :]  # [1, 1, D]
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                # update decoder state and get next score
                y, state, _ = self.decoder.score_hypothesis(max_hyp, cache)  # [1, 1, D]

                # get next token
                ytu = torch.log_softmax(self.joint.joint(hi, y) / self.softmax_temperature, dim=-1)  # [1, 1, 1, V + 1]
                ytu = ytu[0, 0, 0, :]  # [V + 1]

                # remove blank token before top k
                top_k = ytu[ids].topk(beam_k, dim=-1)

                # Two possible steps - blank token or non-blank token predicted
                ytu = (
                    torch.cat((top_k[0], ytu[self.blank].unsqueeze(0))),
                    torch.cat((top_k[1] + index_incr, blank_tensor)),
                )

                # for each possible step
                for logp, k in zip(*ytu):
                    # construct hypothesis for step
                    new_hyp = Hypothesis(
                        score=(max_hyp.score + float(logp)),
                        y_sequence=max_hyp.y_sequence[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                        timestep=max_hyp.timestep[:],
                        length=encoded_lengths,
                    )

                    # if current token is blank, dont update sequence, just store the current hypothesis
                    if k == self.blank:
                        kept_hyps.append(new_hyp)
                    else:
                        # if non-blank token was predicted, update state and sequence and then search more hypothesis
                        new_hyp.dec_state = state
                        new_hyp.y_sequence.append(int(k))
                        new_hyp.timestep.append(i)

                        hyps.append(new_hyp)

                # keep those hypothesis that have scores greater than next search generation
                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted([hyp for hyp in kept_hyps if hyp.score > hyps_max], key=lambda x: x.score,)

                # If enough hypothesis have scores greater than next search generation,
                # stop beam search.
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break
        return self.sort_nbest(kept_hyps), cache

 