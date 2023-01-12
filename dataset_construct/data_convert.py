#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import json
from typing import Dict, List
from tqdm import tqdm
from collections import Counter
from universal_ie.record_schema import RecordSchema
from universal_ie.dataset import Dataset
from universal_ie.ie_format import Sentence
from universal_ie.logger import init_logger
logger = None



def convert_graph(
    output_folder: str,
    datasets: Dict[str, List[Sentence]],
    label_mapper: Dict = None,
):

    def get_label_str(label):
        return label_mapper.get(label.__repr__(), label.__repr__())

    counter = Counter()
    os.makedirs(output_folder, exist_ok=True)


    schema = {}
    for data_type, instance_list in datasets.items():    
        with open(os.path.join(output_folder, f"{data_type}.json"), "w") as output:
            for instance in tqdm(instance_list):
                counter.update([f"{data_type} sent"])

                for entity in instance.entities:
                    if get_label_str(entity.label) not in schema:
                        schema[get_label_str(entity.label)] = set()  

                for relation in instance.relations:
                    if get_label_str(relation.arg1.label) not in schema:
                        schema[get_label_str(relation.arg1.label)] = set() 
                    schema[get_label_str(relation.arg1.label)].add(get_label_str(relation.label)) 

                for event in instance.events:
                    if get_label_str(event.label) not in schema:
                        schema[get_label_str(event.label)] = set() 
                    for arg_role, _ in event.args:
                        schema[get_label_str(event.label)].add(get_label_str(arg_role))

                output.write(
                    "%s\n"
                    % json.dumps(
                        {
                            "text": ' '.join(instance.tokens),
                            "tokens": instance.tokens,
                            "entity": [
                                entity.to_offset(label_mapper)
                                for entity in instance.entities
                            ],
                            "relation": [
                                relation.to_offset(
                                    ent_label_mapper=label_mapper,
                                    rel_label_mapper=label_mapper,
                                )
                                for relation in instance.relations
                            ],
                            "event": [
                                event.to_offset(evt_label_mapper=label_mapper)
                                for event in instance.events
                            ],
                        },
                        ensure_ascii=False,
                    )
                )


    RecordSchema.output_schema(schema, os.path.join(output_folder, "schema.json"))
    logger.info(f"Counter: {dict(counter)}")    
    print(output_folder)
    print("==========================")






def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", default="data_config/entity")
    parser.add_argument("--iter_num", dest="iter_num", default=7, type=int)
    parser.add_argument("--task", dest="task", default="NYT", choices=['NYT', 'Few-NERD', 'ace05_event'])
    parser.add_argument("--mode", dest="mode", default="V")
    options = parser.parse_args()

    global logger
    logger = init_logger(task_name = f"{options.task}_{options.mode}")

    for it in range(1, options.iter_num + 1):
        filename = f"{options.config}/{options.task}_{options.mode}{it}.yaml"
        logger.info(f"Filename: {filename}")  
        dataset = Dataset.load_yaml_file(filename)
        datasets = dataset.load_dataset(logger_name=f"{options.task}_{options.mode}")    
        label_mapper = dataset.mapper
        output_name = f"../data/iter_{it}/{options.task}_{options.mode}"  

        convert_graph(
            output_name,
            datasets=datasets,
            label_mapper=label_mapper,
        )



if __name__ == "__main__":
    main()
