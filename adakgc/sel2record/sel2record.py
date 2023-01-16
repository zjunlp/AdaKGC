#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import json
import logging
from collections import defaultdict, OrderedDict
from adakgc.utils.record_schema import RecordSchema
from adakgc.extraction.predict_parser import SpotAsocPredictParser
from adakgc.sel2record.record import EntityRecord, MapConfig, RelationRecord, EventRecord


logger = logging.getLogger("__main__")


task_record_map = {
    'entity': EntityRecord,
    'relation': RelationRecord,
    'event': EventRecord,
}




def proprocessing_graph_record(graph, schema_dict, task):
    """ Mapping generated spot-asoc result to Entity/Relation/Event
    将抽取的Spot-Asoc结构, 根据不同的 Schema 转换成 Entity/Relation/Event 结果
    """
    records = {
        'entity': list(),
        'relation': list(),
        'event': list(),
    }

    entity_dict = OrderedDict()
    
    for record in graph['pred_record']:
        if record['type'] in schema_dict.type_list:
            if task == "event":
                records['event'] += [{
                    'trigger': record['spot'],
                    'type': record['type'],
                    'roles': record['asocs']
                }]
            else:
                records['entity'] += [{
                    'text': record['spot'],
                    'type': record['type']
                }]
                entity_dict[record['spot']] = record['type']

        else:
            print("Type `%s` invalid." % record['type'])


    for record in graph['pred_record']:
        if record['type'] in schema_dict.type_list:
            for role in record['asocs']:
                records['relation'] += [{
                    'type': role[0],
                    'roles': [
                        (record['type'], record['spot']),
                        (entity_dict.get(role[1], record['type']), role[1]),
                    ]
                }]

    if len(entity_dict) > 0 and task == "event":
        for record in records['event']:
            if record['type'] in schema_dict.type_list:
                new_role_list = list()
                for role in record['roles']:
                    if role[1] in entity_dict:
                        new_role_list += [role]
                record['roles'] = new_role_list

    return records



class SEL2Record:
    def __init__(self, schema_dict, map_config: MapConfig, task) -> None:
        self._schema_dict = schema_dict
        self._predict_parser = SpotAsocPredictParser(
            label_constraint=schema_dict,
        )
        self._map_config = map_config
        self._task = task

    def __repr__(self) -> str:
        return f"## {self._map_config}"



    def sel2record(self, pred, text, tokens):  
        # Parsing generated SEL to String-level Record
        well_formed_list, counter = self._predict_parser.decode(
            gold_list=[],
            pred_list=[pred],
            text_list=[text],
        )

        # Convert String-level Record to Entity/Relation/Event
        pred_records = proprocessing_graph_record(      
            well_formed_list[0],
            self._schema_dict,
            self._task,
        )


        pred = defaultdict(dict)
        # Mapping String-level record to Offset-level record
        for task in task_record_map:
            record_map = task_record_map[task](
                map_config=self._map_config,
            )

            pred[task]['offset'] = record_map.to_offset(
                instance=pred_records.get(task, []),
                token_list=tokens,
            )     

            pred[task]['string'] = record_map.to_string(
                instance=pred_records.get(task, []),
                token_list=tokens,
            )   


        return pred



    @staticmethod
    def load_schema_dict(schema_file):
        lines = open(schema_file).readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        type_role_dict = json.loads(lines[2])
        return RecordSchema(type_list, role_list, type_role_dict)
