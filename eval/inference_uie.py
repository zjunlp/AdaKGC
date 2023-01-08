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
from transformers import T5TokenizerFast, T5ForConditionalGeneration

from uie.extraction.record_schema import RecordSchema
from uie.sel2record.record import MapConfig
from uie.extraction.scorer import *
from uie.sel2record.sel2record import SEL2Record


logger = logging.getLogger(__name__)


split_bracket = re.compile(r"\s*<extra_id_\d>\s*")
special_to_remove = {'<pad>', '</s>'}


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





class HuggingfacePredictor:
    def __init__(self, args) -> None:
        self._tokenizer = T5TokenizerFast.from_pretrained(args.model)
        self._model = T5ForConditionalGeneration.from_pretrained(args.model)
        self._model.cuda(f"cuda:{args.cuda}")     
        self._max_source_length = args.max_source_length
        self._max_target_length = args.max_target_length


    def load_schema(self, record_file): 
        self._schema = RecordSchema.read_from_file(record_file) 
        self._ssi = schema_to_ssi(self._schema)
        logger.info(f"record_file: {record_file}")  
        logger.info(f"ssi: {self._ssi}")


    def predict(self, text):
        text = [self._ssi + x for x in text]          
        inputs = self._tokenizer(text, padding=True, return_tensors='pt').to(self._model.device)
        inputs['input_ids'] = inputs['input_ids'][:, :self._max_source_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :self._max_source_length] 

        result = self._model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self._max_target_length,
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
    parser.add_argument('--data', default='data/relation/NYT')
    parser.add_argument('--model', default='hf_models/t5-v1_1-base')
    parser.add_argument('--task', default='relation')
    parser.add_argument('--mode', default='H')
    parser.add_argument('--cuda', default='0')

    parser.add_argument('--max_source_length', default=256, type=int)
    parser.add_argument('--max_target_length', default=192, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--config', dest='map_config', help='Offset Re-mapping Config', default='config/offset_map/closest_offset_en.yaml')
    parser.add_argument('--decoding', default='spotasoc')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--match_mode', default='normal', choices=['set', 'normal', 'multimatch'])
    

                
    options = parser.parse_args()
    if options.task == "relation":
        tgt = [18, 24, 30, 36]
    elif options.task == "event":
        tgt = [42, 48, 54, 60]
    elif options.task == "entity":
        tgt = [6, 12]


    model_path = '_'.join(options.model.split('/')[1:]).replace('/', '_')
    os.makedirs(f'output_infer/{model_path}', exist_ok = True)
    data_dir = options.data.replace('/', '_')
    output_dir = f'output_infer/{model_path}/{data_dir}'
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, 'test_results.txt')):
        cur_time = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
        output_dir += cur_time
    os.makedirs(output_dir, exist_ok = True)

    logging.basicConfig(
        format="%(asctime)s - %(funcName)s - %(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir+'/log.txt', mode = 'w', encoding = 'utf-8')],
    )
    logger.setLevel(logging.INFO)
    logger.info(f"config: f{vars(options)}")
    logger.info(f"data: {data_dir}")


    predictor = HuggingfacePredictor(args=options) 
    predictor.load_schema(f"{options.data}/record.schema")  
    map_config = MapConfig.load_from_yaml(options.map_config)
    schema_dict = SEL2Record.load_schema_dict(options.data_folder)
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema=options.decoding,
        map_config=map_config,
    )

    for split, split_name in [('test', 'test')]:
        gold_filename = f"{options.data}/{split}.json"
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
                    number.append(line.split("=")[-1])

        for num in number:
            logger.info(f"{num.strip()}")


if __name__ == "__main__":
    main()
