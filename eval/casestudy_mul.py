#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import yaml
import argparse
from uie.extraction.record_schema import RecordSchema
from prompt.sel2record.cases import EntityCase, RelationCase




def read_json_file(file_name):
    return [json.loads(line) for line in open(file_name)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_dir', default='output_infer/ace05_event/baseline-1_ace05_event_R')
    parser.add_argument('--data_name', default='ace05_event') 
    parser.add_argument('--task', default='event') 
    parser.add_argument('--mode', default='R')   
    parser.add_argument('--iter', type=int, default=7) 
    parser.add_argument('--CD', action='store_true')
    options = parser.parse_args()

    gold_files = []
    pred_files = []
    new_lists = []
    basepath = "dataset_processing/converted_data/text2spotasoc"

    if options.task == 'entity':
        task_dict = {
            'entity': EntityCase,
        }
    elif options.task == 'relation':
        task_dict = {
            'relation': RelationCase,
        }
    else:
        task_dict = {
            'event': EventCase,
        }

    for i in range(1, options.iter+1):
        gold_files.append(f'{basepath}/iter_n-{i}/{options.data_name}_{options.mode}')
        pred_files.append(f'{options.infer_dir}/iter_n-{i}_{options.data_name}_{options.mode}')
        new_list_file = f'config/data_config/{options.task}/{options.data_name}_{options.mode}{i}.yaml'
        dataset_config = yaml.load(open(new_list_file), Loader=yaml.FullLoader)
        new_lists.append(dataset_config.get('new_list', []))
        if options.CD:
            gold_files.append(f'{basepath}/iter_n-{i}/{options.data_name}_{options.mode}')
            pred_files.append(f'{options.infer_dir}/iter_n-{i}_{options.data_name}_{options.mode}_CD')
            new_lists.append(dataset_config.get('new_list', []))


    result_logger = open(f'{options.infer_dir}/log.txt', 'w')
    results = {}
    for i in range(len(gold_files)):
        gold_filename = gold_files[i]
        pred_filename = pred_files[i]

        results[pred_filename] = {}
        for task, counter in task_dict.items():
            results[pred_filename][task] = {}

            gold_list = [x[task] for x in read_json_file(gold_filename+'/test.json')]
            pred_list = [x[task] for x in read_json_file(pred_filename+'/test_preds_record.txt')]
            record_schema = RecordSchema.read_from_file(gold_filename+'/record.schema')

            gold_instance_list = counter.load_gold_list(gold_list)
            pred_instance_list = counter.load_pred_list(pred_list)

            result = counter.eval_instance_list(
                gold_instance_list=gold_instance_list,
                pred_instance_list=pred_instance_list,
                output_dir=pred_filename, 
                record_schema=record_schema,
                new_list=new_lists[i], 
            )

            for (eval_type, r) in result:
                results[pred_filename][task][eval_type] = r    
    
    result_logger.write(json.dumps(results)+'\n')
    print(results)


if __name__ == "__main__":
    main()
