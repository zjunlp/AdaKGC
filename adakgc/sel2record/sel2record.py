#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import defaultdict, OrderedDict
import os
from adakgc.utils.record_schema import RecordSchema
from adakgc.extraction.predict_parser import get_predict_parser
from adakgc.sel2record.record import EntityRecord, MapConfig, RelationRecord, EventRecord
import logging

logger = logging.getLogger("__main__")


task_record_map = {
    'entity': EntityRecord,
    'relation': RelationRecord,
    'event': EventRecord,
}




def proprocessing_graph_record(graph, schema_dict):
    """ Mapping generated spot-asoc result to Entity/Relation/Event
    将抽取的Spot-Asoc结构, 根据不同的 Schema 转换成 Entity/Relation/Event 结果
    """
    records = {
        'entity': list(),
        'relation': list(),
        'event': list(),
    }

    entity_dict = OrderedDict()

    # 根据不同任务的 Schema 将不同的 Spot 对应到不同抽取结果： Entity/Event
    # Mapping generated spot result to Entity/Event
    for record in graph['pred_record']:

        if record['type'] in schema_dict['entity'].type_list:
            records['entity'] += [{
                'text': record['spot'],
                'type': record['type']
            }]
            entity_dict[record['spot']] = record['type']

        elif record['type'] in schema_dict['event'].type_list:
            records['event'] += [{
                'trigger': record['spot'],
                'type': record['type'],
                'roles': record['asocs']
            }]

        else:
            print("Type `%s` invalid." % record['type'])

    # 根据不同任务的 Schema 将不同的 Asoc 对应到不同抽取结果： Relation/Argument
    # Mapping generated asoc result to Relation/Argument
    for record in graph['pred_record']:
        if record['type'] in schema_dict['entity'].type_list:
            for role in record['asocs']:
                records['relation'] += [{
                    'type': role[0],
                    'roles': [
                        (record['type'], record['spot']),
                        (entity_dict.get(role[1], record['type']), role[1]),
                    ]
                }]

    if len(entity_dict) > 0:
        for record in records['event']:
            if record['type'] in schema_dict['event'].type_list:
                new_role_list = list()
                for role in record['roles']:
                    if role[1] in entity_dict:
                        new_role_list += [role]
                record['roles'] = new_role_list

    return records


class SEL2Record:
    def __init__(self, schema_dict, decoding_schema, map_config: MapConfig) -> None:
        self._schema_dict = schema_dict
        self._predict_parser = get_predict_parser(
            decoding_schema=decoding_schema,
            label_constraint=schema_dict['record'],
        )
        self._map_config = map_config

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
        pred_records = proprocessing_graph_record(      # 上面decode传入的是[pred]，当然只有1个啊
            well_formed_list[0],
            self._schema_dict
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
    def load_schema_dict(schema_folder):
        schema_dict = dict()
        for schema_key in ['record', 'entity', 'relation', 'event']:
            schema_filename = os.path.join(schema_folder, f'{schema_key}.schema')
            if os.path.exists(schema_filename):
                schema_dict[schema_key] = RecordSchema.read_from_file(
                    schema_filename
                )
            else:
                logger.warning(f"{schema_filename} is empty, ignore.")
                schema_dict[schema_key] = RecordSchema.get_empty_schema()
        return schema_dict
