from __future__ import absolute_import
from operator import itemgetter
import argparse
from dataclasses import dataclass
import numpy as np
import glob
import json
import torch
import os
import nemo.collections.asr as nemo_asr
import nemo
from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
# from nemo.collections.asr.models.conda import EncDecRNNTBPEModel
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.utils import logging, model_utils
import copy
from nemo.collections.asr.metrics.wer import word_error_rate
from omegaconf import OmegaConf


try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


def check_input(args):
    if args.model_path is None and args.pretrained_name is None:
        raise ValueError("Both args.model_path and args.pretrained_name cannot be None!")
    if args.audio_dir is None and args.dataset_manifest is None:
        raise ValueError("Both args.audio_dir and args.dataset_manifest cannot be None!")

def get_device(args):
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_data(args):
    # get audio filenames
    filepaths = []
    gold_texts = []
    if args.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    else:
        # get filenames from manifest
        with open(args.dataset_manifest, 'r') as f:
            for line in f:
                item = json.loads(line)
                filepaths.append(item['audio_filepath'])
                gold_texts.append(item['text'])
    logging.info(f"\nTranscribing {len(filepaths)} files...\n")

    return filepaths, gold_texts


def load_model(args, device):
        # setup model
    torch.set_grad_enabled(False)
    if args.model_path is not None:
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=args.model_path, return_config=True)
        
        # print('model_cfg',model_cfg)
      
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
       
        asr_model = imported_class.restore_from(restore_path=args.model_path, map_location=device)  # type: ASRModel
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(model_name=args.pretrained_name, map_location=device)  # type: ASRModel
        model_name = args.pretrained_name
    
    # asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_medium")
    asr_model = asr_model.eval()
    logging.info(asr_model.cfg.decoding)
    new_decoding = copy.deepcopy(asr_model.cfg.decoding)
    new_decoding.strategy = "greedy"
    # #    new_decoding.beam.beam_size = 1
    # print(new_decoding.beam)
    # if args.search_type == 'greedy':
     
    # elif args.search_type != 'beam':
        # raise ValueError('search type must be beam or greedy')
    asr_model.change_decoding_strategy(new_decoding)

    # logging.info(asr_model.cfg.decoding)
    # asr_model.export('./model.onnx')
    # exit()
    return asr_model, model_name

def infer(args, asr_model, filepaths, logprobs):
    # transcribe audio
    with autocast():
        with torch.no_grad():
            hypotheses = asr_model.transcribe(filepaths, 
                                              batch_size=args.batch_size)#,

            # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
            if type(hypotheses) == tuple and len(hypotheses) == 2:
                hypotheses = hypotheses[0]    
    return hypotheses



def save_transcriptions(args, filepaths, transcriptions):
    with open(args.output_filename, 'w', encoding='utf-8') as f:
        if args.audio_dir is not None:
            for idx, text in enumerate(transcriptions):
                item = {'audio_filepath': filepaths[idx], 'pred_text': text}
                f.write(json.dumps(item) + "\n")
        else:
            with open(args.dataset_manifest, 'r') as fr:
                for idx, line in enumerate(fr):
                    item = json.loads(line)
                    item['pred_text'] = transcriptions[idx]
                    f.write(json.dumps(item) + "\n")

        logging.info("Finished writing predictions !")
        logging.info(f"Saved into {args.output_filename}!")

@dataclass
class NemoClient:

    model_path: str #local model name or checkpoint
    pretrained_name: str #Cloud model name  like stt_en_conformer_ctc_small 
    audio_dir: str #Folder contains audio files
    dataset_manifest: str  #Manifest containing audio paths
    output_filename: str='x'
    search_type: str = 'greedy' #beam or greedy
    lm_path: str='' #lm path
    alpha: float=0.931 
    beta: float=1.183
    beam_width: int=128
    n_cpu: int=8
    use_cpu: bool=False
    batch_size: int=32
    encoding_level: str='char'
    norm_energy: bool=False

    def evaluate(self):
        
        check_input(self)
        device = get_device(self)
        
        asr_model, model_name = load_model(self, device)
        filepaths, gold_texts = get_data(self)

        logprobs = True if self.lm_path else False
        hypotheses = infer(self,
                        asr_model=asr_model, 
                        filepaths=filepaths, 
                        logprobs=logprobs)
        
        if self.lm_path:
            vocab = asr_model.decoder.vocabulary
            ids_to_text_func = None
            if self.encoding_level == "subword":
                TOKEN_OFFSET = 100
                vocab = [chr(idx + TOKEN_OFFSET) for idx in range(len(vocab))]
                ids_to_text_func = asr_model.tokenizer.ids_to_text

            beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
                vocab=vocab,
                beam_width=self.beam_width,
                alpha=self.alpha, 
                beta=self.beta,
                lm_path=self.lm_path,
                num_cpus=self.n_cpu,
                cutoff_top_n=len(vocab),
                cutoff_prob=1,
                input_tensor=False)

            exp_hypotheses = [softmax(i) for i in hypotheses]
            with nemo.core.typecheck.disable_checks():
                beam_search_pred = beam_search_lm.forward(log_probs=exp_hypotheses, log_probs_length=None)
            hypotheses = []
            for i in range(len(exp_hypotheses)):
                best_beam = max(beam_search_pred[i], key=itemgetter(0))[1]
                if ids_to_text_func is not None:
                    # For BPE encodings, need to shift by TOKEN_OFFSET to retrieve the original sub-word ids
                    pred_text = ids_to_text_func([ord(c) - TOKEN_OFFSET for c in best_beam])
                else:
                    pred_text = best_beam
                hypotheses.append(pred_text)
        if self.output_filename:
            save_transcriptions(self, filepaths, hypotheses)
        wer_value = word_error_rate(hypotheses=hypotheses, references=gold_texts)
        print("Model: {},\nDataset: {}\nWER: {}".format(model_name, os.path.split(self.dataset_manifest)[1], wer_value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--pretrained_name", type=str)
    parser.add_argument("--audio_dir", type=str)
    parser.add_argument("--dataset_manifest", type=str)
    parser.add_argument("--output_filename", type=str)
    parser.add_argument("--lm_path", type=str, default='')
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--beam_width", type=int, default=128)
    parser.add_argument("--n_cpu", type=int, default=8)   
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--encoding_level", type=str, default="char")
    parser.add_argument("--norm_energy", action="store_true", default=False)
    
    args = parser.parse_args()
    args.model_path = '/home/tsargsyan/davit/nemo-models/stt_en_conformer_transducer_medium.nemo'
    # args.model_path = 
    args.dataset_manifest = '/home/tsargsyan/saten/NeMo3/toy-manifest.json'
    # args.dataset_manifest = '/data/ASR_DATA/ljspeech/asr-ljspeech-test-textnorm-nonzeros-duration.json'
    client = NemoClient(**vars(args))
    client.evaluate()

if __name__=="__main__":
    main()
