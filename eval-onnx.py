import glob
import json
import os
import tempfile
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models.asr_model import ASRModel
from onnx_decoding import ONNXGreedyBatchedRNNTInfer
from nemo.utils import logging
import onnx
import onnxruntime  



def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--nemo_model", type=str, default=None,  help="Path to .nemo file",
    )
    parser.add_argument('--onnx_encoder', type=str, default=None, required=False, help="Path to onnx encoder model")
    parser.add_argument(
        '--onnx_decoder', type=str, default=None, required=False, help="Path to onnx decoder + joint model"
    )

    parser.add_argument('--threshold', type=float, default=0.01, required=False)

    parser.add_argument('--dataset_manifest', type=str, default=None, required=False, help='Path to dataset manifest')
    parser.add_argument('--audio_dir', type=str, default=None, required=False, help='Path to directory of audio files')
    parser.add_argument('--audio_type', type=str, default='wav', help='File format of audio')

    parser.add_argument('--export', action='store_true', help="Whether to export the model into onnx prior to eval")
    parser.add_argument('--max_symbold_per_step', type=int, default=5, required=False, help='Number of decoding steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batchsize')
    parser.add_argument('--log', action='store_true', help='Log the predictions between pytorch and onnx')

    args = parser.parse_args()
    args.nemo_model = '/home/tsargsyan/davit/nemo-models/stt_en_conformer_transducer_medium.nemo'
    args.onnx_encoder = './onnxdir/Encoder-tranducer.onnx'
    args.onnx_decoder = './onnxdir/Decoder-Joint-tranducer.onnx'
    # args.dataset_manifest = '/data/ASR_DATA/ljspeech/asr-ljspeech-test-textnorm-nonzeros-duration.json'
    args.dataset_manifest = '/home/tsargsyan/saten/NeMo3/toy-manifest.json'
    args.batch_size = 1
    
    return args


def assert_args(args):
    if args.nemo_model is None:
        raise ValueError(
            "`nemo_model` must be passed ! It is required for decoding the RNNT tokens and ensuring predictions "
            "match between Torch and ONNX."
        )

    if args.export and (args.onnx_encoder is not None or args.onnx_decoder is not None):
        raise ValueError("If `export` is set, then `onnx_encoder` and `onnx_decoder` arguments must be None")

    if args.audio_dir is None and args.dataset_manifest is None:
        raise ValueError("Both `dataset_manifest` and `audio_dir` cannot be None!")

    if args.audio_dir is not None and args.dataset_manifest is not None:
        raise ValueError("Submit either `dataset_manifest` or `audio_dir`.")

    if int(args.max_symbold_per_step) < 1:
        raise ValueError("`max_symbold_per_step` must be an integer > 0")


def export_model_if_required(args, nemo_model):
    if args.export:
        nemo_model.export("temp_rnnt.onnx")
        args.onnx_encoder = "Encoder-temp_rnnt.onnx"
        args.onnx_decoder = "Decoder-Joint-temp_rnnt.onnx"

def resolve_audio_filepaths(args):
    # get audio filenames
    if args.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(args.audio_dir.audio_dir, f"*.{args.audio_type}")))
    else:
        # get filenames from manifest
        filepaths = []
        gold_texts = []
        with open(args.dataset_manifest, 'r') as f:
            for line in f:
                item = json.loads(line)
                filepaths.append(item['audio_filepath'])
                gold_texts.append(item['text'])

    logging.info(f"\nTranscribing {len(filepaths)} files...\n")

    return filepaths, gold_texts



class Client:
    def __init__(self, encoder_onnx_path, past_m = 980, present_n = 1000, future_k = 1020):
            # initializing sreaming params
        assert past_m + present_n + future_k >900, 'this sum past_m + present_n + future_k should be > 900'
        assert past_m % 40 == 20 , 'past_m % 40 == 20 should hold'
        assert present_n % 40 == 0 , 'present_n % 40 == 0 should hold'
        assert future_k % 10 ==0, 'future_k%10 ==0'

        self.past_m_samples = past_m * 16 # samplerate / 1000
        self.present_n_samples = present_n *16
        self.future_k_samples = future_k *16

        self.encoded_start = past_m // 40 +1
        self.encoded_end = (past_m + present_n) // 40 + 1

        print('encoded_start',self.encoded_start)
        print('encoded_end',self.encoded_end)

        new_encoded_len = present_n // 40
        self.new_encoded_len = torch.tensor([new_encoded_len])
        self.chunk_len =  self.past_m_samples + self.present_n_samples + self.future_k_samples
           
            # initializing onnx encoder
        self.initialize_encoder(encoder_onnx_path)
        assert hasattr(self, 'encoder'), 'error'


    def run_encoder(self, audio_signal, length):
        # print('audio_signal.shape',audio_signal.shape)
        # print('length',length)
        if hasattr(audio_signal, 'cpu'):
            audio_signal = audio_signal.detach().cpu().numpy()

        if hasattr(length, 'cpu'):
            length = length.detach().cpu().numpy()
        # print('length.shape',length.shape)
        # print('length',length)
        ip = {
            'audio_signal': audio_signal,
            'length': length,
        }
        enc_out = self.encoder.run(None, ip)
        enc_out, encoded_length = enc_out  # ASSUME: single output
        # print('enc_out.shape',enc_out.shape)
        # print('encoded_length',encoded_length)
        return enc_out, encoded_length
   
    def evaluate(self):
        args = parse_arguments()

        #streaming params
        

        # Instantiate pytorch model
        nemo_model = args.nemo_model
        nemo_model = ASRModel.restore_from(nemo_model, map_location='cpu')  # type: ASRModel
        nemo_model.freeze()

        if torch.cuda.is_available():
            nemo_model = nemo_model.to('cuda')

        export_model_if_required(args, nemo_model)

        # Instantiate RNNT Decoding loop
        encoder_model = args.onnx_encoder
        decoder_model = args.onnx_decoder
        max_symbols_per_step = args.max_symbold_per_step
        decoding = ONNXGreedyBatchedRNNTInfer(encoder_model, decoder_model, max_symbols_per_step)
        audio_filepath, gold_texts = resolve_audio_filepaths(args)
        
        # Evaluate Pytorch Model (CPU/GPU)
        actual_transcripts = nemo_model.transcribe(audio_filepath, batch_size=args.batch_size)[0]
        print('actual_transcripts',actual_transcripts)
        # Evaluate ONNX model (on CPU)
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                for audio_file in audio_filepath:
                    entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                    fp.write(json.dumps(entry) + '\n')

            config = {'paths2audio_files': audio_filepath, 'batch_size': args.batch_size, 'temp_dir': tmpdir}

            # Push nemo model to CPU
            nemo_model = nemo_model.to('cpu')
            nemo_model.preprocessor.featurizer.dither = 0.0
            nemo_model.preprocessor.featurizer.pad_to = 0

            temporary_datalayer = nemo_model._setup_transcribe_dataloader(config)

            all_hypothesis = []
            for test_batch in tqdm(temporary_datalayer, desc="ONNX Transcribing"):
                
                input_signal, input_signal_length = test_batch[0], test_batch[1]
                input_signal, input_signal_length = self.pad_audio(input_signal, input_signal_length)

                processed_audio, processed_audio_len = nemo_model.preprocessor(
                    input_signal=input_signal, length=input_signal_length
                )
            
                
                # RNNT Decoding loop
                hypotheses = decoding(audio_signal=processed_audio, length=processed_audio_len)
                # Process hypothesis (map char/subword token ids to text)
                hypotheses = nemo_model.decoding.decode_hypothesis(hypotheses)  # type: List[str]

                # Extract text from the hypothesis
                texts = [h.text for h in hypotheses]

                all_hypothesis += texts
                del processed_audio, processed_audio_len
                del test_batch

        if args.log:
            for pt_transcript, onnx_transcript in zip(actual_transcripts, all_hypothesis):
                print(f"Pytorch Transcripts : {pt_transcript}")
                print(f"ONNX Transcripts    : {onnx_transcript}")
            print()

        # Measure error rate between onnx and pytorch transcipts
        pt_onnx_cer = word_error_rate(all_hypothesis, gold_texts, use_cer=True)
        assert pt_onnx_cer < args.threshold, "Threshold violation !"

        print("Character error rate between Pytorch and ONNX :", pt_onnx_cer)

    def initialize_encoder(self, onnx_path):
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model, full_check=True)
        self.encoder = onnxruntime.InferenceSession(onnx_model.SerializeToString())

    def pad_audio(self, audio, len_audio):
        pad2 = self.present_n_samples - len_audio % self.present_n_samples
        start_padding = torch.zeros(audio.shape[0],self.past_m_samples)
        end_padding = torch.zeros(audio.shape[0],self.future_k_samples+pad2)
        audio = torch.cat((start_padding, audio, end_padding), dim = 1)
        len_audio = len_audio + self.past_m_samples + self.future_k_samples + pad2
        return audio, len_audio

def main():
    #streaming params
    args = parse_arguments()
    client = Client(encoder_onnx_path = args.onnx_encoder)
    client.evaluate()

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter


