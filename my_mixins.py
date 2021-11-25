
import os
from abc import ABC, abstractmethod
from typing import List

from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.collections.common import tokenizers
from nemo.utils import logging


class ASRBPEMixin(ABC):
    """ ASR BPE Mixin class that sets up a Tokenizer via a config

    This mixin class adds the method `_setup_tokenizer(...)`, which can be used by ASR models
    which depend on subword tokenization.

    The setup_tokenizer method adds the following parameters to the class -
        -   tokenizer_cfg: The resolved config supplied to the tokenizer (with `dir` and `type` arguments).
        -   tokenizer_dir: The directory path to the tokenizer vocabulary + additional metadata.
        -   tokenizer_type: The type of the tokenizer. Currently supports `bpe` and `wpe`.
        -   vocab_path: Resolved path to the vocabulary text file.

    In addition to these variables, the method will also instantiate and preserve a tokenizer
    (subclass of TokenizerSpec) if successful, and assign it to self.tokenizer.
    """

    def _setup_tokenizer(self, tokenizer_cfg: DictConfig, 
                        tokenizer_model_path,
                        tokenizer_vocab_path):
        # Prevent tokenizer parallelism (unless user has explicitly set it)
        if 'TOKENIZERS_PARALLELISM' not in os.environ:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.tokenizer_cfg = OmegaConf.to_container(tokenizer_cfg, resolve=True)  # type: dict
        self.tokenizer_dir = self.tokenizer_cfg.pop('dir')  # Remove tokenizer directory
        self.tokenizer_type = self.tokenizer_cfg.pop('type').lower()  # Remove tokenizer_type

        self.hf_tokenizer_kwargs = self.tokenizer_cfg.pop("hf_kwargs", {})  # Remove HF tokenizer kwargs

        # Preserve config
       
        if hasattr(self, 'cfg') and 'tokenizer' in self.cfg:
            self.cfg.tokenizer.dir = self.tokenizer_dir
            self.cfg.tokenizer.type = self.tokenizer_type

            if 'hf_kwargs' in tokenizer_cfg:
                with open_dict(self.cfg.tokenizer):
                    self.cfg.tokenizer.hf_kwargs = tokenizer_cfg.get('hf_kwargs')

       
        assert self.tokenizer_type == 'bpe', 'error'
            # This is a BPE Tokenizer
        if 'special_tokens' in self.tokenizer_cfg:
            special_tokens = self.tokenizer_cfg['special_tokens']

            if special_tokens is not None:
                raise ValueError("`special_tokens` are no longer supported for SentencePiece based tokenizers.")

        # Update special tokens
        self.tokenizer = tokenizers.SentencePieceTokenizer(model_path=tokenizer_model_path)
        self.vocab_path = tokenizer_vocab_path

        vocabulary = {}
        for i in range(self.tokenizer.vocab_size):
            piece = self.tokenizer.ids_to_tokens([i])
            piece = piece[0]
            vocabulary[piece] = i + 1

        # wrapper method to get vocabulary conveniently
        def get_vocab():
            return vocabulary

        # attach utility values to the tokenizer wrapper
        self.tokenizer.tokenizer.vocab_size = len(vocabulary)
        self.tokenizer.tokenizer.get_vocab = get_vocab
        self.tokenizer.tokenizer.all_special_tokens = self.tokenizer.special_token_to_id

        