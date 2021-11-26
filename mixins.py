
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import numpy as np
import sentencepiece
from omegaconf import DictConfig, OmegaConf, open_dict


__all__ =['ASRBPEMixin', 'SentencePieceTokenizer', 'TokenizerSpec' ]


class TokenizerSpec(ABC):
    """
    Inherit this class to implement a new tokenizer.
    """

    @abstractmethod
    def text_to_tokens(self, text):
        pass

    @abstractmethod
    def tokens_to_text(self, tokens):
        pass

    @abstractmethod
    def tokens_to_ids(self, tokens):
        pass

    @abstractmethod
    def ids_to_tokens(self, ids):
        pass

    @abstractmethod
    def text_to_ids(self, text):
        pass

    @abstractmethod
    def ids_to_text(self, ids):
        pass

    def add_special_tokens(self, special_tokens: List[str]):
        raise NotImplementedError("To be implemented")

    @property
    def name(self):
        return type(self).__name__

class SentencePieceTokenizer(TokenizerSpec):
    '''
    Sentencepiecetokenizer https://github.com/google/sentencepiece.
        Args:
        model_path: path to sentence piece tokenizer model. To create the model use create_spt_model()
        special_tokens: either list of special tokens or dictionary of token name to token value
        legacy: when set to True, the previous behavior of the SentecePiece wrapper will be restored, 
            including the possibility to add special tokens inside wrapper.
    '''

    def __init__(
        self, model_path: str, special_tokens: Optional[Union[Dict[str, str], List[str]]] = None, legacy: bool = False
    ):
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"model_path: {model_path} is invalid")
        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.Load(model_path)

        self.original_vocab_size = self.tokenizer.get_piece_size()
        self.vocab_size = self.tokenizer.get_piece_size()
        self.legacy = legacy
        self.special_token_to_id = {}
        self.id_to_special_token = {}
        if special_tokens:
            if not self.legacy:
                raise ValueError(
                    "Special tokens must be None when legacy is set to False. Provide special tokens at train time."
                )
            self.add_special_tokens(special_tokens)

    def text_to_tokens(self, text):
        if self.legacy:
            tokens = []
            idx = 0
            last_idx = 0

            while 1:
                indices = {}

                for token in self.special_token_to_id:
                    try:
                        indices[token] = text[idx:].index(token)
                    except ValueError:
                        continue

                if len(indices) == 0:
                    break

                next_token = min(indices, key=indices.get)
                next_idx = idx + indices[next_token]

                tokens.extend(self.tokenizer.encode_as_pieces(text[idx:next_idx]))
                tokens.append(next_token)
                idx = next_idx + len(next_token)

            tokens.extend(self.tokenizer.encode_as_pieces(text[idx:]))
            return tokens

        return self.tokenizer.encode_as_pieces(text)

    def text_to_ids(self, text):
        if self.legacy:
            ids = []
            idx = 0
            last_idx = 0

            while 1:
                indices = {}

                for token in self.special_token_to_id:
                    try:
                        indices[token] = text[idx:].index(token)
                    except ValueError:
                        continue

                if len(indices) == 0:
                    break

                next_token = min(indices, key=indices.get)
                next_idx = idx + indices[next_token]

                ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
                ids.append(self.special_token_to_id[next_token])
                idx = next_idx + len(next_token)

            ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
            return ids

        return self.tokenizer.encode_as_ids(text)

    def tokens_to_text(self, tokens):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        return self.tokenizer.decode_pieces(tokens)

    def ids_to_text(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()

        if self.legacy:
            text = ""
            last_i = 0

            for i, id in enumerate(ids):
                if id in self.id_to_special_token:
                    text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                    text += self.id_to_special_token[id] + " "
                    last_i = i + 1

            text += self.tokenizer.decode_ids(ids[last_i:])
            return text.strip()

        return self.tokenizer.decode_ids(ids)

    def token_to_id(self, token):
        if self.legacy and token in self.special_token_to_id:
            return self.special_token_to_id[token]

        return self.tokenizer.piece_to_id(token)

    def ids_to_tokens(self, ids):
        tokens = []
        for id in ids:
            if id >= self.original_vocab_size:
                tokens.append(self.id_to_special_token[id])
            else:
                tokens.append(self.tokenizer.id_to_piece(id))
        return tokens

    def tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = []
        for token in tokens:
            ids.append(self.token_to_id(token))
        return ids

    def add_special_tokens(self, special_tokens):
        if not self.legacy:
            raise AttributeError("Special Token addition does not work when legacy is set to False.")

        if isinstance(special_tokens, list):
            for token in special_tokens:
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1
        elif isinstance(special_tokens, dict):
            for token_name, token in special_tokens.items():
                setattr(self, token_name, token)
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1

    @property
    def pad_id(self):
        if self.legacy:
            pad_id = self.tokens_to_ids([self.pad_token])[0]
        else:
            pad_id = self.tokenizer.pad_id()
        return pad_id

    @property
    def bos_id(self):
        if self.legacy:
            bos_id = self.tokens_to_ids([self.bos_token])[0]
        else:
            bos_id = self.tokenizer.bos_id()
        return bos_id

    @property
    def eos_id(self):
        if self.legacy:
            eos_id = self.tokens_to_ids([self.eos_token])[0]
        else:
            eos_id = self.tokenizer.eos_id()
        return eos_id

    @property
    def sep_id(self):
        if self.legacy:
            return self.tokens_to_ids([self.sep_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def cls_id(self):
        if self.legacy:
            return self.tokens_to_ids([self.cls_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def unk_id(self):
        return self.tokenizer.unk_id()

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
        self.tokenizer = SentencePieceTokenizer(model_path=tokenizer_model_path)
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



