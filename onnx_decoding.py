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

class ONNXGreedyBatchedRNNTInfer:
    def __init__(
        self, encoder_model: str, decoder_joint_model: str, max_symbols_per_step: Optional[int] = None,
    ):
        try:
            import onnx
            import onnxruntime
        except (ModuleNotFoundError, ImportError):
            raise ImportError(f"`onnx` or `onnxruntime` could not be imported, please install the libraries.\n")



        onnx_model = onnx.load(encoder_model)
        onnx.checker.check_model(onnx_model, full_check=True)
        self.encoder_model = onnx_model
        self.encoder = onnxruntime.InferenceSession(onnx_model.SerializeToString())

        onnx_model = onnx.load(decoder_joint_model)
        onnx.checker.check_model(onnx_model, full_check=True)
        self.decoder_joint_model = onnx_model
        self.decoder_joint = onnxruntime.InferenceSession(onnx_model.SerializeToString())

        logging.info("Successfully loaded encoder, decoder and joint onnx models !")

        # Will be populated at runtime
        self._blank_index = None
        self.max_symbols_per_step = max_symbols_per_step

        self._setup_encoder_input_output_keys()
        self._setup_decoder_joint_input_output_keys()
        self._setup_blank_index()
       

    def _setup_encoder_input_output_keys(self):
        self.encoder_inputs = list(self.encoder_model.graph.input)
        self.encoder_outputs = list(self.encoder_model.graph.output)
       
    def _setup_decoder_joint_input_output_keys(self):
        self.decoder_joint_inputs = list(self.decoder_joint_model.graph.input)
        self.decoder_joint_outputs = list(self.decoder_joint_model.graph.output)

    def _setup_blank_index(self):
        # ASSUME: Single input with no time length information
        dynamic_dim = 257
        shapes = self.encoder_inputs[0].type.tensor_type.shape.dim
        ip_shape = []

        for shape in shapes:
            if hasattr(shape, 'dim_param') and 'dynamic' in shape.dim_param:
                ip_shape.append(dynamic_dim)  # replace dynamic axes with constant
            else:
                ip_shape.append(int(shape.dim_value))
                # torch.Size([1, 154481])
        # test_batch.shape torch.Size([1])
        # print('torch.randint(0, 1, size=(dynamic_dim,)',torch.Size([1])
        # print('torch.randn(*ip_shape).shape',torch.randn(torch.Size([1, 154481]).shape)

        # self.run_encoder(
        #     audio_signal = torch.randn(*[1, 80, 257]),length=torch.randint(0, 1, size=(257,))
        # )
        
        enc_logits, encoded_length = self.run_encoder(
            audio_signal=torch.randn(*ip_shape), length=torch.randint(0, 1, size=(dynamic_dim,))
        )

        # prepare states
        states = self._get_initial_states(batchsize=dynamic_dim)

        # run decoder 1 step
        joint_out, states = self.run_decoder_joint(enc_logits, None, None, *states)
        log_probs, lengths = joint_out

        self._blank_index = log_probs.shape[-1] - 1  # last token of vocab size is blank token
        logging.info(
            f"Enc-Dec-Joint step was evaluated, blank token id = {self._blank_index}; vocab size = {log_probs.shape[-1]}"
        )

    

    def __call__(self, audio_signal: torch.Tensor, length: torch.Tensor):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        with torch.no_grad():
         
            # Apply optional preprocessing
            encoder_output, encoded_lengths = self.run_encoder(audio_signal=audio_signal, length=length)
            encoder_output = encoder_output.transpose([0, 2, 1])  # (B, T, D)
            logitlen = encoded_lengths

            inseq = encoder_output  # [B, T, D]
            hypotheses, timestamps = self._greedy_decode(inseq, logitlen)

            # Pack the hypotheses results
            packed_result = [rnnt_utils.Hypothesis(score=-1.0, y_sequence=[]) for _ in range(len(hypotheses))]
            for i in range(len(packed_result)):
                packed_result[i].y_sequence = torch.tensor(hypotheses[i], dtype=torch.long)
                packed_result[i].length = timestamps[i]

            del hypotheses

        return packed_result

    def _greedy_decode(self, x, out_len):
        # x: [B, T, D]
        # out_len: [B]

        # Initialize state
        batchsize = x.shape[0]
        # exit('batchsize'+str(batchsize))
        hidden = self._get_initial_states(batchsize)
        target_lengths = torch.ones(batchsize, dtype=torch.int32)

        # Output string buffer
        label = [[] for _ in range(batchsize)]
        timesteps = [[] for _ in range(batchsize)]

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long).numpy()
        print('last_label.shape',last_label.shape)
        e
        # Mask buffers
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool).numpy()

        # Get max sequence length
        max_out_len = out_len.max()
        for time_idx in range(max_out_len):
            f = x[:, time_idx : time_idx + 1, :]  # [B, 1, D]
            f = f.transpose([0, 2, 1])

            # Prepare t timestamp batch variables
            not_blank = True
            symbols_added = 0

            # Reset blank mask
            blank_mask *= False

            # Update blank mask with time mask
            # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
            # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
            blank_mask = time_idx >= out_len
            # Start inner loop
            while not_blank and (self.max_symbols_per_step is None or symbols_added < self.max_symbols_per_step):

                # Batch prediction and joint network steps
                # If very first prediction step, submit SOS tag (blank) to pred_step.
                # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                if time_idx == 0 and symbols_added == 0:
                    g = torch.tensor([self._blank_index] * batchsize, dtype=torch.int32).view(-1, 1)
                else:
                    g = last_label.astype(np.int32)

                # Batched joint step - Output = [B, V + 1]
                joint_out, hidden_prime = self.run_decoder_joint(f, g, target_lengths, *hidden)
                logp, pred_lengths = joint_out
                logp = logp[:, 0, 0, :]

                # Get index k, of max prob for batch
                k = np.argmax(logp, axis=1).astype(np.int32)

                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                k_is_blank = k == self._blank_index
                blank_mask |= k_is_blank

                del k_is_blank
                del logp

                # If all samples predict / have predicted prior blanks, exit loop early
                # This is equivalent to if single sample predicted k
                if blank_mask.all():
                    not_blank = False

                else:
                    # Collect batch indices where blanks occurred now/past
                    blank_indices = blank_mask.astype(np.int32).nonzero()
                    if type(blank_indices) in (list, tuple):
                        blank_indices = blank_indices[0]

                    # Recover prior state for all samples which predicted blank now/past
                    if hidden is not None:
                        # LSTM has 2 states
                        for state_id in range(len(hidden)):
                            hidden_prime[state_id][:, blank_indices, :] = hidden[state_id][:, blank_indices, :]

                    elif len(blank_indices) > 0 and hidden is None:
                        # Reset state if there were some blank and other non-blank predictions in batch
                        # Original state is filled with zeros so we just multiply
                        # LSTM has 2 states
                        for state_id in range(len(hidden_prime)):
                            hidden_prime[state_id][:, blank_indices, :] *= 0.0

                    # Recover prior predicted label for all samples which predicted blank now/past
                    k[blank_indices] = last_label[blank_indices, 0]

                    # Update new label and hidden state for next iteration
                    last_label = k.copy().reshape(-1, 1)
                    hidden = hidden_prime

                    # Update predicted labels, accounting for time mask
                    # If blank was predicted even once, now or in the past,
                    # Force the current predicted label to also be blank
                    # This ensures that blanks propogate across all timesteps
                    # once they have occured (normally stopping condition of sample level loop).
                    for kidx, ki in enumerate(k):
                        if blank_mask[kidx] == 0:
                            label[kidx].append(ki)
                            timesteps[kidx].append(time_idx)

                    symbols_added += 1

        return label, timesteps

    def run_encoder(self, audio_signal, length):
        # print('audio_signal.shape',audio_signal.shape)
        # print('length',length)
        if hasattr(audio_signal, 'cpu'):
            audio_signal = audio_signal.detach().cpu().numpy()

        if hasattr(length, 'cpu'):
            length = length.detach().cpu().numpy()
        # print('length.shape',length.shape)
        # print('length',length)
        # print('self.encoder_inputs[0].name',self.encoder_inputs[0].name)
        # print('self.encoder_inputs[1].name',self.encoder_inputs[1].name)
        # exit('exiting2')
        ip = {
            self.encoder_inputs[0].name: audio_signal,
            self.encoder_inputs[1].name: length,
        }
        enc_out = self.encoder.run(None, ip)
        enc_out, encoded_length = enc_out  # ASSUME: single output
        # print('enc_out.shape',enc_out.shape)
        # print('encoded_length',encoded_length)
        return enc_out, encoded_length

    def run_decoder_joint(self, enc_logits, targets, target_length, *states):
        # ASSUME: Decoder is RNN Transducer
        if targets is None:
            targets = torch.zeros(enc_logits.shape[0], 1, dtype=torch.int32)
            target_length = torch.ones(enc_logits.shape[0], dtype=torch.int32)

        if hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()

        if hasattr(target_length, 'cpu'):
            target_length = target_length.cpu().numpy()

        ip = {
            self.decoder_joint_inputs[0].name: enc_logits,
            self.decoder_joint_inputs[1].name: targets,
            self.decoder_joint_inputs[2].name: target_length,
        }

        num_states = 0
        if states is not None and len(states) > 0:
            num_states = len(states)
            for idx, state in enumerate(states):
                if hasattr(state, 'cpu'):
                    state = state.cpu().numpy()

                ip[self.decoder_joint_inputs[len(ip)].name] = state

        dec_out = self.decoder_joint.run(None, ip)

        # unpack dec output
        if num_states > 0:
            new_states = dec_out[-num_states:]
            dec_out = dec_out[:-num_states]
        else:
            new_states = None

        return dec_out, new_states

    def _get_initial_states(self, batchsize):
        # ASSUME: LSTM STATES of shape (layers, batchsize, dim)
        input_state_nodes = [ip for ip in self.decoder_joint_inputs if 'state' in ip.name]
        num_states = len(input_state_nodes)
        if num_states == 0:
            return

        input_states = []
        for state_id in range(num_states):
            node = input_state_nodes[state_id]
            ip_shape = []
            for shape_idx, shape in enumerate(node.type.tensor_type.shape.dim):
                if hasattr(shape, 'dim_param') and 'dynamic' in shape.dim_param:
                    ip_shape.append(batchsize)  # replace dynamic axes with constant
                else:
                    ip_shape.append(int(shape.dim_value))

            input_states.append(torch.zeros(*ip_shape))

        return input_states