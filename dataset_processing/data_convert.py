#!/usr/bin/env python
# -*- coding:utf-8 -*-
from asyncio.log import logger
from collections import Counter
import os
import json
from typing import Dict, List
from tqdm import tqdm
from universal_ie.generation_format.generation_format import GenerationFormat
from universal_ie.generation_format import generation_format_dict
from universal_ie.generation_format.structure_marker import BaseStructureMarker
from universal_ie.dataset import Dataset
from universal_ie.ie_format import Sentence
from universal_ie.logger import init_logger
logger = None


def convert_graph(
    generation_class: GenerationFormat,
    output_folder: str,
    datasets: Dict[str, List[Sentence]],
    language: str = "en",
    label_mapper: Dict = None,
):
    convertor = generation_class(
        structure_maker=BaseStructureMarker(),
        language=language,
        label_mapper=label_mapper,
    )

    counter = Counter()
    os.makedirs(output_folder, exist_ok=True)

    schema_counter = {
        "entity": list(),
        "relation": list(),
        "event": list(),
    }
    for data_type, instance_list in datasets.items():    
        with open(os.path.join(output_folder, f"{data_type}.json"), "w") as output:
            for instance in tqdm(instance_list):
                counter.update([f"{data_type} sent"])
                converted_graph = convertor.annonote_graph(
                    tokens=instance.tokens,
                    entities=instance.entities,
                    relations=instance.relations,
                    events=instance.events,
                )
                src, tgt, spot_labels, asoc_labels = converted_graph[:4]
                spot_asoc = converted_graph[4]

                schema_counter["entity"] += instance.entities
                schema_counter["relation"] += instance.relations
                schema_counter["event"] += instance.events

                output.write(
                    "%s\n"
                    % json.dumps(
                        {
                            "text": src,
                            "tokens": instance.tokens,
                            "record": tgt,
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
                            "spot": list(spot_labels),
                            "asoc": list(asoc_labels),
                            "spot_asoc": spot_asoc,
                        },
                        ensure_ascii=False,
                    )
                )
    convertor.output_schema(os.path.join(output_folder, "schema.json"))
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

    generation_class = generation_format_dict.get('spotasoc')  
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
            generation_class,
            output_name,
            datasets=datasets,
            language=dataset.language,
            label_mapper=label_mapper,
        )



if __name__ == "__main__":
    main()
