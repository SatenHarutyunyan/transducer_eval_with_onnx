

from typing import List, Optional

class CharParser:
    """Functor for parsing raw strings into list of int tokens.

    Examples:
        >>> parser = CharParser(['a', 'b', 'c'])
        >>> parser('abc')
        [0, 1, 2]
    """

    def __init__(
        self,
        labels: List[str],
        *,
        unk_id: int = -1,
        blank_id: int = -1,
        do_normalize: bool = True,
        do_lowercase: bool = True,
        do_tokenize: bool = True,
    ):
        """Creates simple mapping char parser.

        Args:
            labels: List of labels to allocate indexes for. Essentially,
                this is a id to str mapping.
            unk_id: Index to choose for OOV words (default: -1).
            blank_id: Index to filter out from final list of tokens
                (default: -1).
            do_normalize: True if apply normalization step before tokenizing
                (default: True).
            do_lowercase: True if apply lowercasing at normalizing step
                (default: True).
        """

        self._labels = labels
        self._unk_id = unk_id
        self._blank_id = blank_id
        self._do_normalize = do_normalize
        self._do_lowercase = do_lowercase
        self._do_tokenize = do_tokenize

        self._labels_map = {label: index for index, label in enumerate(labels)}
        self._special_labels = set([label for label in labels if len(label) > 1])

    def __call__(self, text: str) -> Optional[List[int]]:
        if self._do_normalize:
            text = self._normalize(text)
            if text is None:
                return None

        if not self._do_tokenize:
            return text

        text_tokens = self._tokenize(text)
        return text_tokens

    def _normalize(self, text: str) -> Optional[str]:
        text = text.strip()

        if self._do_lowercase:
            text = text.lower()

        return text

    def _tokenize(self, text: str) -> List[int]:
        tokens = []
        # Split by word for find special labels.
        for word_id, word in enumerate(text.split(' ')):
            if word_id != 0:  # Not first word - so we insert space before.
                tokens.append(self._labels_map.get(' ', self._unk_id))

            if word in self._special_labels:
                tokens.append(self._labels_map[word])
                continue

            for char in word:
                tokens.append(self._labels_map.get(char, self._unk_id))

        # If unk_id == blank_id, OOV tokens are removed.
        tokens = [token for token in tokens if token != self._blank_id]

        return tokens

    def decode(self, str_input):
        r_map = {}
        for k, v in self._labels_map.items():
            r_map[v] = k
        r_map[len(self._labels_map)] = "<BOS>"
        r_map[len(self._labels_map) + 1] = "<EOS>"
        r_map[len(self._labels_map) + 2] = "<P>"

        out = []
        for i in str_input:
            # Skip OOV
            if i not in r_map:
                continue
            out.append(r_map[i.item()])

        return "".join(out)
