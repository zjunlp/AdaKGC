#!/usr/bin/env python
# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
import os
import argparse
import logging
import json
import time
import re
from tqdm import tqdm
import sys

import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration


from uie.seq2seq.constraint_decoder import get_constraint_decoder
from uie.seq2seq.models import T5Prompt, T5Prefix

from uie.sel2record.record import MapConfig
from uie.sel2record.sel2record import SEL2Record

from uie.extraction.scorer import *
from uie.extraction.record_schema import RecordSchema
from uie.extraction.constants import type_start, type_end, span_start, null_span


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
    def __init__(self, decoding_format = 'spotasoc', source_prefix = '', args = None) -> None:
        self._tokenizer = T5TokenizerFast.from_pretrained(args.model)
        logger.info(f"Tokenizer Length: {len(self._tokenizer)}")
        self._device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self._device}")
        self._model = T5Prompt(self._tokenizer, args.t5_path, args).to(self._device)
        self._model.load_state_dict(torch.load(os.path.join(args.model, 'pytorch_model.bin'), map_location=self._device))
        self._model.eval()

        self._max_source_length = args.max_source_length
        self._max_target_length = args.max_target_length
        self._use_ssi = args.use_ssi
        self._args = {"num_beams": args.num_beams, "do_sample": args.do_sample, "top_k": args.top_k, "top_p": args.top_p}
        self.task_name = args.task


    def load_schema(self, record_file, CD):
        logger.info(f"record_file: {record_file}")
        self._schema = RecordSchema.read_from_file(record_file) 
        spots, asocs = schema_to_spotasoc(self._schema, self._tokenizer)
        self._ssi = schema_to_ssi(self._schema)
        self._spots = spots
        self._asocs = asocs
        logger.info(f"ssi: {self._ssi}")
        logger.info(f"spots: {self._spots}")
        logger.info(f"asocs: {self._asocs}")
        if CD:
            self.constraint_decoder = get_constraint_decoder(tokenizer = self._tokenizer,
                                                             type_schema = self._schema,
                                                             decoding_schema = 'spotasoc',
                                                             source_prefix = '',
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
            **self._args
        )

        return self._tokenizer.batch_decode(result, skip_special_tokens=False, clean_up_tokenization_spaces=False)



class HuggingfacePredictor:
    def __init__(self, decoding_format = 'spotasoc', source_prefix = '', args = None) -> None:
        self._tokenizer = T5TokenizerFast.from_pretrained(args.model)
        self._model = T5ForConditionalGeneration.from_pretrained(args.model)
        self._model.cuda(f"cuda:{args.cuda}")
                 
        self._max_source_length = args.max_source_length
        self._max_target_length = args.max_target_length
        self._args = {"num_beams": args.num_beams, "do_sample": args.do_sample, "top_k": args.top_k, "top_p": args.top_p}
        self.task_name = args.task


    def load_schema(self, record_file, CD): 
        logger.info(f"record_file: {record_file}")  
        self._schema = RecordSchema.read_from_file(record_file) 
        self._ssi = schema_to_ssi(self._schema)
        logger.info(f"ssi: {self._ssi}")
        if CD:
            self.constraint_decoder = get_constraint_decoder(tokenizer = self._tokenizer,
                                                             type_schema = self._schema,
                                                             decoding_schema = 'spotasoc',
                                                             source_prefix = '',
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

        text = [self._ssi + x for x in text]          # SSI作前缀
        inputs = self._tokenizer(text, padding=True, return_tensors='pt').to(self._model.device)
        inputs['input_ids'] = inputs['input_ids'][:, :self._max_source_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :self._max_source_length] 

        result = self._model.generate(
            input_ids=inputs['input_ids'],
            prefix_allowed_tokens_fn=func,
            attention_mask=inputs['attention_mask'],
            max_length=self._max_target_length,
            **self._args
        )
        
        return self._tokenizer.batch_decode(result, skip_special_tokens=False, clean_up_tokenization_spaces=False)



task_dict = {
    'entity': EntityScorer,
    'relation': RelationScorer,
    'event': EventScorer,
}



def do_predict(predictor, output_dir, split_name, batch_num, options, text_list):
    predicts = list()
    if os.path.exists(os.path.join(output_dir, f'{split_name}_preds_seq2seq.txt')):
        with open(os.path.join(output_dir, f'{split_name}_preds_seq2seq.txt'), 'r') as reader:
            for line in reader:
                predicts.append(line.strip())
        return predicts
    
    for index in tqdm(range(batch_num)):
        start = index * options.batch_size
        end = index * options.batch_size + options.batch_size

        pred_seq2seq = predictor.predict(text_list[start: end])
        pred_seq2seq = [post_processing(x) for x in pred_seq2seq]

        predicts += pred_seq2seq

    with open(os.path.join(output_dir, f'{split_name}_preds_seq2seq.txt'), 'w') as output:
        for pred in predicts:
            output.write(f'{pred}\n')

    return predicts


def do_sel2record(predicts, sel2record, text_list, token_list, output_dir, split_name):
    records = list()
    if os.path.exists(os.path.join(output_dir, f'{split_name}_preds_record.txt')):
        with open(os.path.join(output_dir, f'{split_name}_preds_record.txt'), 'r') as reader:
            for line in reader:
                records.append(json.loads(line.strip()))
        return records

    for p, text, tokens in zip(predicts, text_list, token_list):
        r = sel2record.sel2record(pred=p, text=text, tokens=tokens)
        records += [r]

    with open(os.path.join(output_dir, f'{split_name}_preds_record.txt'), 'w') as output:
        for record in records:
            output.write(f'{json.dumps(record)}\n')
    
    return records


 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='relation/NYT')
    parser.add_argument('--model', default='hf_models/mix')
    parser.add_argument('--task', default='relation')
    parser.add_argument('--iter_num', default=7, type=int)
    parser.add_argument('--cuda', default='0')
    parser.add_argument('--t5_path', default='hf_models/mix', type=str)

    parser.add_argument('--max_source_length', default=256, type=int)
    parser.add_argument('--max_target_length', default=192, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--config', dest='map_config', help='Offset Re-mapping Config', default='config/offset_map/closest_offset_en.yaml')
    parser.add_argument('--decoding', default='spotasoc')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--match_mode', default='set', choices=['set', 'normal', 'multimatch'])
    parser.add_argument('--mode', default='H')

    parser.add_argument('--CD', action='store_true')
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--use_prefix', action='store_true')
    parser.add_argument('--use_ssi', action='store_true')
    parser.add_argument('--prompt_len', default=10, type=int)
    parser.add_argument('--prompt_dim', default=512, type=int)

    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--num_beams', default=None, type=int)
    parser.add_argument('--top_k', default=None, type=int)
    parser.add_argument('--top_p', default=None, type=float)
    
    options = parser.parse_args()
    map_config = MapConfig.load_from_yaml(options.map_config)

    # only F1 value
    if options.task == "relation":
        tgt = [18, 24, 30, 36]
    elif options.task == "event":
        tgt = [42, 48, 54, 60]
    elif options.task == "entity":
        tgt = [6, 12]


    if options.use_prompt:
        predictor = HuggingfacePromptPredictor(args=options) 
    else:
        predictor = HuggingfacePredictor(args=options) 



    data_folder = []
    for it in range(1, options.iter_num + 1):
        data_folder.append(os.path.join(options.dataname, f"/iter_{it}"))
    data_folder = sorted(data_folder, key=lambda x:x)
    print(data_folder)
    
    number_dict = {}
    for data in data_folder:
        options.data_folder = data
        model_path = '_'.join(options.model.split('/')[1:]).replace('/', '_')
        if options.num_beams != None:
            model_path += f'_beam{options.num_beams}'
        if options.do_sample:
            if options.top_k != None:
                model_path += f'_topk{options.top_k}'
            if options.top_p != None:
                model_path += f'_topp{options.top_p}'
        os.makedirs(os.path.join('output_infer', model_path), exist_ok = True)
        
        
        data_dir = data.replace('/', '_')
        output_dir = os.path.join('output_infer', model_path, data_dir)
        if options.CD:
            output_dir += '_CD'
        if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, 'test_results.txt')):
            cur_time = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
            output_dir += cur_time
        os.makedirs(output_dir, exist_ok = True)
        
        logging.basicConfig(
            format="%(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join('output_infer', 'log.txt'), mode = 'w', encoding = 'utf-8')],
        )
        logger.setLevel(logging.INFO)
        logger.info(f"config: {vars(options)}")
        logger.info(f"data: {data}")


        predictor.load_schema(os.path.join('output_infer', 'record.schema'), options.CD)   
        schema_dict = SEL2Record.load_schema_dict(data)
        sel2record = SEL2Record(
            schema_dict=schema_dict,
            decoding_schema=options.decoding,
            map_config=map_config,
        )


        for split, split_name in [('test', 'test')]:
            gold_filename = os.path.join(data, f'{split}.json')

            text_list = [x['text'] for x in read_json_file(gold_filename)]
            token_list = [x['tokens'] for x in read_json_file(gold_filename)]

            batch_num = math.ceil(len(text_list) / options.batch_size)

            predicts = do_predict(predictor, output_dir, split_name, batch_num, options, text_list)
            records = do_sel2record(predicts, sel2record, text_list, token_list, output_dir, split_name)

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
            number_dict[data] = number

            for num in number:
                logger.info(f"{num}")


    for key, value in number_dict.items():
        logger.info(key)
        for it in value:
            logger.info(it)

            

    


if __name__ == "__main__":
    main()
