from abc import ABC, abstractmethod
import torch
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
from torch.nn import Module

__all__ = ['NeuralModule']


class NeuralModule(Module):
    """
    Abstract class offering interface shared between all PyTorch Neural Modules.
    """

    def freeze(self) -> None:
        r"""
        Freeze all params for inference.
        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

  

class AudioPreprocessor(NeuralModule, ABC):
    """
        An interface for Neural Modules that performs audio pre-processing,
        transforming the wav files to features.
    """

    def __init__(self, win_length, hop_length):
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length

        self.torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }

   
    @torch.no_grad()
    def forward(self, input_signal, length):
        processed_signal, processed_length = self.get_features(input_signal, length)

        return processed_signal, processed_length

    @abstractmethod
    def get_features(self, input_signal, length):
        # Called by forward(). Subclasses should implement this.
        pass


class AudioToMelSpectrogramPreprocessor(AudioPreprocessor):

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(
                tuple('B'), LengthsType()
            ),  # Please note that length should be in samples not seconds.
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(MelSpectrogramSignalTag)
            2: AxisType(ProcessedTimeTag)
        processed_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        features=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=1e-5,
        pad_to=16,
        frame_splicing=1,
        exact_pad=False,
        stft_exact_pad=False,
        stft_conv=False,
        pad_value=0,
        mag_power=2.0,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
    ):
        super().__init__(n_window_size, n_window_stride)

        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        self.featurizer = FilterbankFeatures(
            sample_rate=self._sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            exact_pad=exact_pad,
            stft_exact_pad=stft_exact_pad,
            stft_conv=stft_conv,
            pad_value=pad_value,
            mag_power=mag_power,
            rng=rng,
            nb_augmentation_prob=nb_augmentation_prob,
            nb_max_freq=nb_max_freq,
        )

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length)

    @property
    def filter_banks(self):
        return self.featurizer.filter_banks
