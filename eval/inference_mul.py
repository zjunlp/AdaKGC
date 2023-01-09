#!/usr/bin/env python
# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
import os
import sys
sys.path.append('./')
import argparse
import logging
import json
import time
import re
from tqdm import tqdm

import torch
from transformers import T5TokenizerFast

from adakgc.utils.record_schema import RecordSchema
from adakgc.extraction.scorer import *
from adakgc.sel2record.record import MapConfig
from adakgc.sel2record.sel2record import SEL2Record
from adakgc.models.models import T5Prompt
from adakgc.models import get_constraint_decoder


logger = logging.getLogger(__name__)

split_bracket = re.compile(r"\s*<extra_id_\d>\s*")
special_to_remove = {'<pad>', '</s>'}
cwd = os.getcwd()

def read_json_file(file_name):
    return [json.loads(line) for line in open(file_name)]


def schema_to_ssi(schema: RecordSchema):
    ssi = "<spot> " + " <spot> ".join(sorted(schema.type_list))
    ssi += "<asoc> " + " <asoc> ".join(sorted(schema.role_list))
    ssi += " <extra_id_2> "
    return ssi



def post_processing(x):
    for special in special_to_remove:
        x = x.replace(special, '')
    return x.strip()


def schema_to_spotasoc(schema: RecordSchema, tokenizer):
    spots = []
    asocs = []
    for spot in sorted(schema.type_list):
        spots.append(tokenizer.encode(spot, add_special_tokens = False))
    for asoc in sorted(schema.role_list):
        asocs.append(tokenizer.encode(asoc, add_special_tokens = False))
    return spots, asocs



class HuggingfacePromptPredictor:
    def __init__(self, args) -> None:
        self._tokenizer = T5TokenizerFast.from_pretrained(args.model)
        self._device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        self._model = T5Prompt(self._tokenizer, args.t5_path, args).to(self._device)
        self._model.load_state_dict(torch.load(os.path.join(args.model, 'pytorch_model.bin'), map_location=self._device))
        self._model.eval()

        self._max_source_length = args.max_source_length
        self._max_target_length = args.max_target_length
        self._use_ssi = args.use_ssi
        self.task_name = args.task

        
    def load_schema(self, record_file, CD):
        self._schema = RecordSchema.read_from_file(record_file) 
        spots, asocs = schema_to_spotasoc(self._schema, self._tokenizer)
        self._ssi = schema_to_ssi(self._schema)
        self._spots = spots
        self._asocs = asocs
        logger.info(f"record_file: {record_file}")
        logger.info(f"ssi: {self._ssi}")
        logger.info(f"spots: {self._spots}")
        logger.info(f"asocs: {self._asocs}")
        if CD:
            self.constraint_decoder = get_constraint_decoder(tokenizer = self._tokenizer,
                                                             record_schema = self._schema,
                                                             decoding_schema = 'spotasoc',
                                                             task_name = self.task_name)
        else:
            self.constraint_decoder = None
            

    def predict(self, text):
        func = None
        def CD_fn(batch_id, sent):
            src_sentence = inputs['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence = src_sentence, tgt_generated = sent)
        if self.constraint_decoder is not None:
            func = CD_fn
        
        if self._use_ssi:
            text = [self._ssi + x for x in text]  
        inputs = self._tokenizer(text, padding=True, return_tensors='pt').to(self._device)
        inputs['input_ids'] = inputs['input_ids'][:, :self._max_source_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :self._max_source_length] 

        result = self._model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            spot=[self._spots] * inputs["input_ids"].size(0),
            asoc=[self._asocs] * inputs["input_ids"].size(0),
            prefix_allowed_tokens_fn=func,
        )

        return self._tokenizer.batch_decode(result, skip_special_tokens=False, clean_up_tokenization_spaces=False)




def do_predict(predictor, text_list, file_name, batch_num, options):
    predicts = list()
    if os.path.exists(file_name):
        with open(file_name, 'r') as reader:
            for line in reader:
                predicts.append(line.strip())
        return predicts
    
    for index in tqdm(range(batch_num)):
        start = index * options.batch_size
        end = index * options.batch_size + options.batch_size

        pred_seq2seq = predictor.predict(text_list[start: end])
        pred_seq2seq = [post_processing(x) for x in pred_seq2seq]

        predicts += pred_seq2seq

    with open(file_name, 'w') as output:
        for pred in predicts:
            output.write(f'{pred}\n')

    return predicts


def do_sel2record(predicts, text_list, token_list, sel2record, file_name):
    records = list()
    if os.path.exists(file_name):
        with open(file_name, 'r') as reader:
            for line in reader:
                records.append(json.loads(line.strip()))
        return records

    for p, text, tokens in zip(predicts, text_list, token_list):
        r = sel2record.sel2record(pred=p, text=text, tokens=tokens)
        records += [r]

    with open(file_name, 'w') as output:
        for record in records:
            output.write(f'{json.dumps(record)}\n')
    
    return records




task_dict = {
    'entity': EntityScorer,
    'relation': RelationScorer,
    'event': EventScorer,
}

 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='NYT')
    parser.add_argument('--model', default='output/NYT_H_1')
    parser.add_argument('--task', default='relation')
    parser.add_argument('--iter_num', default=7, type=int)
    parser.add_argument('--mode', default='H', type=str)
    parser.add_argument('--cuda', default='0')
    parser.add_argument('--t5_path', default='hf_models/mix')

    parser.add_argument('--max_source_length', default=256, type=int)
    parser.add_argument('--max_target_length', default=192, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--config', dest='map_config', help='Offset Re-mapping Config', default='config/offset_map/closest_offset_en.yaml')
    parser.add_argument('--decoding', default='spotasoc')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--match_mode', default='normal', choices=['set', 'normal', 'multimatch'])

    parser.add_argument('--CD', action='store_true')
    parser.add_argument('--use_ssi', action='store_true')
    parser.add_argument('--use_task', action='store_true')
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--prompt_len', default=10, type=int)
    parser.add_argument('--prompt_dim', default=512, type=int)
    

    options = parser.parse_args()
    map_config = MapConfig.load_from_yaml(options.map_config)
    predictor = HuggingfacePromptPredictor(args=options) 

    # only F1 value
    if options.task == "relation":
        tgt = [18, 24, 30, 36]
    elif options.task == "event":
        tgt = [42, 48, 54, 60]
    elif options.task == "entity":
        tgt = [6, 12]


    model_path = '_'.join(options.model.split('/')[1:]).replace('/', '_')
    os.makedirs(f'output_infer/{model_path}', exist_ok = True)
    logging.basicConfig(
        format="%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'output_infer/{model_path}'+'/log.txt', mode = 'w', encoding = 'utf-8')],
    )
    logger.setLevel(logging.INFO)
    logger.info(f"config: {vars(options)}")

    data_folder = []
    for it in range(1, options.iter_num + 1):
        data_folder.append(f"data/iter_{it}/{options.dataname}_{options.mode}")



    for data in data_folder:
        data_dir = data.replace('/', '_')
        output_dir = f'output_infer/{model_path}/{data_dir}'
        if options.CD:
            output_dir += '_CD'
        if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, 'test_results.txt')):
            cur_time = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
            output_dir += cur_time
        os.makedirs(output_dir, exist_ok = True)
        logger.info(f"data: {data}")


        predictor.load_schema(f"{data}/record.schema", options.CD)  
        schema_dict = SEL2Record.load_schema_dict(data)
        sel2record = SEL2Record(
            schema_dict=schema_dict,
            decoding_schema=options.decoding,
            map_config=map_config,
        )


        for split, split_name in [('test', 'test')]:
            gold_filename = f"{data}/{split}.json"

            text_list = [x['text'] for x in read_json_file(gold_filename)]
            token_list = [x['tokens'] for x in read_json_file(gold_filename)]

            batch_num = math.ceil(len(text_list) / options.batch_size)
            seq2seq_file = os.path.join(output_dir, f'{split_name}_preds_seq2seq.txt')
            record_file = os.path.join(output_dir, f'{split_name}_preds_record.txt')

            predicts = do_predict(predictor, text_list, seq2seq_file, batch_num, options)
            records = do_sel2record(predicts, text_list, token_list, sel2record, record_file)


            results = dict()
            for task, scorer in task_dict.items():

                gold_list = [x[task] for x in read_json_file(gold_filename)]
                pred_list = [x[task] for x in records]

                gold_instance_list = scorer.load_gold_list(gold_list)
                pred_instance_list = scorer.load_pred_list(pred_list)

                sub_results = scorer.eval_instance_list(
                    gold_instance_list=gold_instance_list,
                    pred_instance_list=pred_instance_list,
                    verbose=options.verbose,
                    match_mode=options.match_mode,
                )
                results.update(sub_results)


            with open(os.path.join(output_dir, f'{split_name}_results.txt'), 'w') as output:
                for key, value in results.items():
                    output.write(f'{split_name}_{key}={value}\n')


            number = []
            with open(os.path.join(output_dir, f'{split_name}_results.txt'), 'r') as freader:
                for i, line in enumerate(freader, 1):
                    if i in tgt:
                        logger.info(f"{line.strip()}")
                        number.append(line.strip().split("=")[-1])

            for num in number:
                logger.info(f"{num}")


if __name__ == "__main__":
    main()
