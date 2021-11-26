from nemo.collections.asr.data.audio_to_text import AudioToBPEDataset

def get_bpe_dataset(
    config: dict, tokenizer: 'TokenizerSpec'
) -> AudioToBPEDataset:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based AudioToBPEDataset.

    Args:
        config: Config of the AudioToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToBPEDataset.
    """
    augmentor = None
    dataset = AudioToBPEDataset(
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
    )
    return dataset

