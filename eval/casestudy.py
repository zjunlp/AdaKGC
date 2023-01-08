#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import argparse
from uie.extraction.record_schema import RecordSchema
from prompt.sel2record.cases import EntityCase, RelationCase




def read_json_file(file_name):
    return [json.loads(line) for line in open(file_name)]


task_dict = {
    'entity': EntityCase,
    'relation': RelationCase
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_folder', default='event/oneie_ace05_en_event')
    parser.add_argument(
        '--output_dir', default='output_infer/baseline-1_ace05_event_H/oneie_ace05_en_event')
    options = parser.parse_args()
    options.data_folder = f"data/text2spotasoc/{options.data_folder}"

    
    result_logger = open(f'{options.output_dir}/static_result.txt', 'w')
    results = {}
    special_results = {}
    for split, split_name in [('test', 'test')]:
        record_schema = RecordSchema.read_from_file(f"{options.data_folder}/record.schema")
        gold_filename = f"{options.data_folder}/{split}.json"
        pred_filename = f"{options.output_dir}/{split_name}_preds_record.txt"
        
        results[options.output_dir] = {}
        special_results[options.output_dir] = {}
        for task, counter in task_dict.items():
            results[options.output_dir][task] = {}
            special_results[options.output_dir][task] = {}
            gold_list = [x[task] for x in read_json_file(gold_filename)]
            pred_list = [x[task] for x in read_json_file(pred_filename)]

            gold_instance_list = counter.load_gold_list(gold_list)
            pred_instance_list = counter.load_pred_list(pred_list)

            result, special_result = counter.eval_instance_list(
                gold_instance_list=gold_instance_list,
                pred_instance_list=pred_instance_list,
                output_dir=options.output_dir, 
                record_schema=record_schema,
            )

            for (eval_type, r) in result:
                results[options.output_dir][task][eval_type] = r
            for (eval_type, r) in special_result:
                special_results[options.output_dir][task][eval_type] = r
            
    
    result_logger.write(json.dumps(results)+'\n')
    result_logger.write(json.dumps(special_results)+'\n')
    print(results)
    print(special_results)

if __name__ == "__main__":
    main()
