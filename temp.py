import glob 
import os
import json
dataset_manifest = '/data/ASR_DATA/ljspeech/asr-ljspeech-test-textnorm-nonzeros-duration.json'
output_filename = './10toy-manifest.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    with open(dataset_manifest, 'r') as fr:
        i = 0
        for line in fr:
            i+=1
            if i > 10:
                break
            item = json.loads(line)
            # item['pred_text'] = transcriptions[idx]
            f.write(json.dumps(item) + "\n")

