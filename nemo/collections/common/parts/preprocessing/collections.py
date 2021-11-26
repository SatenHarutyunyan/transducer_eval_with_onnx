
import collections
import os
from typing import Any, Dict, List, Optional, Union

from nemo.collections.common.parts.preprocessing import manifest, parsers


class _Collection(collections.UserList):
    """List of parsed and preprocessed data."""

    OUTPUT_TYPE = None  # Single element output type.


class AudioText(_Collection):
    """List of audio-transcript text correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(
        typename='AudioTextEntity', field_names='id audio_file duration text_tokens offset text_raw speaker orig_sr',
    )

    def __init__(
        self,
        ids: List[int],
        audio_files: List[str],
        durations: List[float],
        texts: List[str],
        offsets: List[str],
        speakers: List[Optional[int]],
        orig_sampling_rates: List[Optional[int]],
        parser: parsers.CharParser,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio-text manifest with filters and preprocessing.

        Args:
            ids: List of examples positions.
            audio_files: List of audio files.
            durations: List of float durations.
            texts: List of raw text transcripts.
            offsets: List of duration offsets or None.
            speakers: List of optional speakers ids.
            orig_sampling_rates: List of original sampling rates of audio files.
            parser: Instance of `CharParser` to convert string to tokens.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration. Not compatible with index_by_file_id.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        output_type = self.OUTPUT_TYPE
        data, duration_filtered, num_filtered, total_duration = [], 0.0, 0, 0.0
        if index_by_file_id:
            self.mapping = {}

        for id_, audio_file, duration, offset, text, speaker, orig_sr in zip(
            ids, audio_files, durations, offsets, texts, speakers, orig_sampling_rates
        ):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            text_tokens = parser(text)
            if text_tokens is None:
                duration_filtered += duration
                num_filtered += 1
                continue

            total_duration += duration

            data.append(output_type(id_, audio_file, duration, text_tokens, offset, text, speaker, orig_sr))
            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping[file_id] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                print("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        print("Dataset loaded with",len(data)," files totalling",total_duration / 3600," hours" )
        # print("%d files were filtered totalling %.2f hours" %d num_filtered %f duration_filtered / 3600)

        super().__init__(data)


class ASRAudioText(AudioText):
    """`AudioText` collector from asr structured json files."""

    def __init__(self, manifests_files: Union[str, List[str]], *args, **kwargs):
        """Parse lists of audio files, durations and transcripts texts.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `AudioText` constructor.
            **kwargs: Kwargs to pass to `AudioText` constructor.
        """

        ids, audio_files, durations, texts, offsets, speakers, orig_srs = [], [], [], [], [], [], []
        for item in manifest.item_iter(manifests_files):
            ids.append(item['id'])
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            texts.append(item['text'])
            offsets.append(item['offset'])
            speakers.append(item['speaker'])
            orig_srs.append(item['orig_sr'])

        super().__init__(ids, audio_files, durations, texts, offsets, speakers, orig_srs, *args, **kwargs)
