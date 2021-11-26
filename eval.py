
import glob
import json
import os
import tempfile
from argparse import ArgumentParser
from nemo.collections.asr.data import audio_to_text_dataset

import torch
from tqdm import tqdm
from rnnt_wer_bpe import RNNTBPEDecoding, word_error_rate
from beam_decoding import BeamRNNTInfer
from greedy_decoding import GreedyRNNTInfer

from omegaconf import DictConfig, OmegaConf
import onnx
import onnxruntime  
from mixins import ASRBPEMixin
from serialization import Serialization

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--nemo_model", type=str, default=None,  help="Path to .nemo file")
    parser.add_argument('--onnx_encoder', type=str, default=None, help="Path to onnx encoder model")
    parser.add_argument('--onnx_decoder', type=str, default=None, help="Path to onnx decoder + joint model")
    parser.add_argument('--dataset_manifest', type=str, default=None, help='Path to dataset manifest')
    parser.add_argument('--audio_dir', type=str, default=None, help='Path to directory of audio files')
    parser.add_argument('--audio_type', type=str, default='wav', help='File format of audio')
    parser.add_argument('--max_symbols_per_step', type=int, default=5, help='Number of decoding steps')
    parser.add_argument('--batch_size', type=int, default=1, help='Batchsize')
    parser.add_argument('--output_filename', type=str, default=None)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--decoding_strategy', type=str)

    args = parser.parse_args()
    args.nemo_model = '/home/tsargsyan/davit/nemo-models/stt_en_conformer_transducer_medium.nemo'
    args.onnx_encoder = '../models/onnxdir/Encoder-tranducer.onnx'
    args.decoder_path = '../models/decoder2.pt'
    args.joint_path = '../models/joint2.pt'
    args.config_path = '../models/config/model_config.yaml' 
    # args.dataset_manifest = '/data/ASR_DATA/ljspeech/asr-ljspeech-test-textnorm-nonzeros-duration.json'
    args.dataset_manifest = '../manifests/10toy-manifest.json'
    args.batch_size = 1
    args.decoding_strategy = 'beam'
    args.output_filename = args.dataset_manifest.replace('.json', '-prediction-'+args.decoding_strategy+'.json')
    args.tokenizer_model_path = '/home/tsargsyan/saten/streaming/models/config/e7a9ab91030245cd9d84ef93a02cec35_tokenizer.model'
    args.tokenizer_vocab_path = '/home/tsargsyan/saten/streaming/models/config/6adfc91e052e4ea6886b21f6b2c47b1d_tokenizer.vocab'
    return args


def save_transcriptions(args, filepaths, transcriptions):
    with open(args.output_filename, 'w', encoding='utf-8') as f:
        if args.audio_dir is not None:
            for idx, text in enumerate(transcriptions):
                item = {'audio_filepath': filepaths[idx], 'pred_text': text}
                f.write(json.dumps(item) + "\n")
        else:
            with open(args.dataset_manifest, 'r') as fr:
                for idx, line in enumerate(fr):
                    # item = json.loads(line)
                    # item['pred_text'] = transcriptions[idx]
                    a = {}
                    a['pred_text'] = transcriptions[idx]
                    f.write(json.dumps(a) + "\n")

        print("Finished writing predictions !")
        print("Saved into ",args.output_filename)


def assert_args(args):
    if args.nemo_model is None:
        raise ValueError(
            "`nemo_model` must be passed ! It is required for decoding the RNNT tokens and ensuring predictions "
            "match between Torch and ONNX."
        )

    if args.audio_dir is None and args.dataset_manifest is None:
        raise ValueError("Both `dataset_manifest` and `audio_dir` cannot be None!")

    if args.audio_dir is not None and args.dataset_manifest is not None:
        raise ValueError("Submit either `dataset_manifest` or `audio_dir`.")

    if int(args.max_symbold_per_step) < 1:
        raise ValueError("`max_symbold_per_step` must be an integer > 0")


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

    print("\nTranscribing ", len(filepaths)," files...\n")

    return filepaths, gold_texts

def get_config(config_path: str):
    # can be str path or OmegaConf / DictConfig object
    conf = OmegaConf.load(config_path)
    return conf


class Client:

    def initialize_streaming_params(self, past_m, present_n, future_k):
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

    def initialize_preprocessor(self):
        serial = Serialization()
        self.preprocessor = serial.from_config_dict(self.cfg.preprocessor)
        self.preprocessor.featurizer.dither = 0.0
        self.preprocessor.featurizer.pad_to = 0
 
    def encode(self,current_audio,current_audio_len):
        processed_audio, processed_audio_len = self.preprocessor(input_signal=current_audio, length=current_audio_len)
        encoded, encoded_len = self.run_encoder(processed_audio, processed_audio_len)
        return encoded, encoded_len

    def run_encoder(self, audio_signal, length):
        # print('audio_signal.shape',audio_signal.shape)
        # print('length',length)
        if hasattr(audio_signal, 'cpu'):
            audio_signal = audio_signal.detach().cpu().numpy()

        if hasattr(length, 'cpu'):
            length = length.detach().cpu().numpy()
  
        ip = {
            'audio_signal': audio_signal,
            'length': length,
        }
        enc_out = self.encoder.run(None, ip)
        enc_out, encoded_length = enc_out  # ASSUME: single output
      
        return enc_out, encoded_length   
    
    def initialize_decoder_joint(self, decoder_path, joint_path):
        
        # model = nemo_model.joint  #this is how I save those (classes have torch.nn.module as a grandgrandparent)
        # torch.save(model.state_dict(), './joint.pt')
        # exit('exiting')
        serial = Serialization()
        self.decoder = serial.from_config_dict(self.cfg.decoder)
        self.decoder.load_state_dict(torch.load(decoder_path))  

        self.joint = serial.from_config_dict(self.cfg.joint)
        self.joint.load_state_dict(torch.load(joint_path))  

        # self.decoder = nemo_model.decoding.decoding.decoder
        # self.joint =nemo_model.decoding.decoding.joint

    def initialize_config(self,config_path):
        self.cfg = get_config(config_path) 

   
    def initialize_decoding_and_tokenizer(self,tokenizer_model_path,tokenizer_vocab_path):
        
        mixin = ASRBPEMixin()
        mixin._setup_tokenizer(self.cfg.tokenizer,tokenizer_model_path, tokenizer_vocab_path)
        self.tokenizer = mixin.tokenizer
        self.decoding = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer
        )
        
    def _setup_dataloader_from_config(self, config):

        shuffle = config['shuffle']

        # Instantiate tarred dataset loader or normal dataset loader
        assert not config.get('is_tarred', False), 'error'
        assert not ('manifest_filepath' in config and config['manifest_filepath'] is None),'error'

        dataset = audio_to_text_dataset.get_bpe_dataset(
            config=config, tokenizer=self.tokenizer)

        collate_fn = dataset.collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_transcribe_dataloader(self, config) -> 'torch.utils.data.DataLoader':
        # configs type was Dict from typing

        batch_size = min(config['batch_size'], len(config['paths2audio_files']))
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': min(batch_size, os.cpu_count() - 1),
            'pin_memory': True,
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
        }
        # print('dl_config',dl_config)
        # print("MANE MANE MANEAAAAAAEEEE"*77)
        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer
    
        

    def __init__(self,  encoder_onnx_path, decoder_path, joint_path,
                        nemo_model_path,
                        decoding_strategy,
                        config_path,
                        tokenizer_vocab_path, tokenizer_model_path,
                        past_m = 980, present_n = 1000, future_k = 1020,
                        max_symbols_per_step = None):

            # initializing sreaming params
        
        self.initialize_streaming_params(past_m, present_n, future_k)
        # self.initialize_nemo_model(nemo_model_path) #using for decoding
        self.initialize_config(config_path)
        self.initialize_encoder(encoder_onnx_path)
        self.initialize_decoder_joint(decoder_path, joint_path)
        self.initialize_preprocessor()

        # model_path /tmp/tmptaslozqy/e7a9ab91030245cd9d84ef93a02cec35_tokenizer.model

        # this is not being used in infer
        # set up decoding class
        self.initialize_decoding_and_tokenizer(tokenizer_model_path = tokenizer_model_path, tokenizer_vocab_path = tokenizer_vocab_path)
   
        if decoding_strategy =='beam':
            self.rnnt_infer = BeamRNNTInfer(
                decoder_model=self.decoder,
                joint_model=self.joint,
                beam_size=2, #self.cfg.beam.beam_size,
                search_type='default',
                score_norm=self.cfg.decoding.beam.get('score_norm', True),
                softmax_temperature= 1.0           
            )
        elif decoding_strategy == "greedy":
            self.rnnt_infer = GreedyRNNTInfer(
                    decoder_model=self.decoder,
                    joint_model=self.joint,
                    blank_index=1024,
                    max_symbols_per_step=max_symbols_per_step
                )
        else:
            raise ValueError("decoding strategy must be greedy or beam")

    def infer(self, audio, len_audio):
        """gets padded audio"""
        audio_start = 0
        previous_kept_hyps_and_maybe_cache = None
        while(True):
            audio_end = audio_start + self.chunk_len
            # print('audio_start',audio_start)
            # print('audio_end',audio_end)
            # print('len_audio',len_audio)            
            if audio_end > len_audio:
                break   
            
            current_audio = audio[:,audio_start:audio_end] 
            current_audio_len = torch.tensor([current_audio.shape[1]])

            encoded, encoded_len = self.encode(current_audio,current_audio_len)
            new_encoded = encoded[:,:,self.encoded_start:self.encoded_end]

            best_hyp, nbest_hyps_and_maybe_cache =  self.rnnt_infer(new_encoded,self.new_encoded_len,
                                    previous_kept_hyps_and_maybe_cache=previous_kept_hyps_and_maybe_cache)
            # print('best_hyp',best_hyp)
            hypotheses = self.decoding.decode_hypothesis([best_hyp])  

            # Extract text from the hypothesis
            texts = [h.text for h in hypotheses]
            # print(texts)
            # exit('exiting in infer')
            previous_kept_hyps_and_maybe_cache = nbest_hyps_and_maybe_cache
            audio_start += self.present_n_samples

            del encoded
            del new_encoded
        # print('texts',texts)
        # print('\n \n \n')
        return texts

    def evaluate(self):
        args = parse_arguments()
        # Instantiate pytorch model
       
    
        # decoding = ONNXGreedyBatchedRNNTInfer(encoder_model, decoder_model, max_symbols_per_step)
        audio_filepath, gold_texts = resolve_audio_filepaths(args)
        
        # Evaluate ONNX model (on CPU)
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                for audio_file in audio_filepath:
                    entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                    fp.write(json.dumps(entry) + '\n')

            config = {'paths2audio_files': audio_filepath, 'batch_size': args.batch_size, 'temp_dir': tmpdir}
            # temporary_datalayer = self.nemo_model._setup_transcribe_dataloader(config)
            temporary_datalayer = self._setup_transcribe_dataloader(config)
            all_hypothesis = []
            for test_batch in tqdm(temporary_datalayer, desc="ONNX Transcribing"):
                
                input_signal, input_signal_length = test_batch[0], test_batch[1]
                input_signal, input_signal_length = self.pad_audio(input_signal, input_signal_length)

                texts = self.infer(input_signal, input_signal_length[0])
                # exit('in 249, saten.py exiting')
                
                all_hypothesis += texts
                del input_signal, input_signal_length
                del test_batch
        if args.output_filename:
            save_transcriptions(args, audio_filepath, all_hypothesis)
        # Measure error rate between onnx and pytorch transcipts
        wer = word_error_rate(all_hypothesis, gold_texts, use_cer=False)
        # assert wer < args.threshold, "Threshold violation !"

        print("Word error rate :", wer)

    

def main():
    #streaming params
    args = parse_arguments()
    client = Client(encoder_onnx_path = args.onnx_encoder,   
                    decoder_path = args.decoder_path ,
                    joint_path = args.joint_path ,
                    nemo_model_path = args.nemo_model,
                    config_path = args.config_path,
                    max_symbols_per_step = args.max_symbols_per_step,
                    tokenizer_model_path = args.tokenizer_model_path ,tokenizer_vocab_path = args.tokenizer_vocab_path,
                    decoding_strategy= args.decoding_strategy)
    client.evaluate()

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter


