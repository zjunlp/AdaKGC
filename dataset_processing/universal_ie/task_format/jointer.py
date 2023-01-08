#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
from typing import Counter, List, Mapping
import logging

from universal_ie.utils import tokens_to_str, change_ptb_token_back
from universal_ie.ie_format import Entity, Label, Relation, Sentence, Span
from universal_ie.task_format.task_format import TaskFormat


class JointER(TaskFormat):
    """ Joint Entity Relation Data format at https://github.com/yubowen-ph/JointER"""

    def __init__(self, sentence_json, language='en'):
        super().__init__(
            language=language
        )
        self.tokens = sentence_json['tokens']
        for index in range(len(self.tokens)):
            self.tokens[index] = change_ptb_token_back(self.tokens[index])
        if self.tokens is None:
            print('[sentence without tokens]:', sentence_json)
            exit(1)
        self.spo_list = sentence_json['spo_list']
        self.spo_details = sentence_json['spo_details']
        self.pos_tags = sentence_json['pos_tags']

    def generate_instance(self, delete_list):
        entities = dict()
        relations = dict()
        entity_map = dict()
        counter_entity = Counter()
        counter_relation = Counter()

        for spo_index, spo in enumerate(self.spo_details):
            s_s, s_e, s_t = spo[0], spo[1], spo[2]
            tokens = self.tokens[s_s: s_e]
            indexes = list(range(s_s, s_e))
            if (s_s, s_e, s_t) not in entity_map:
                entities[(s_s, s_e, s_t)] = Entity(       
                    span=Span(
                        tokens=tokens,
                        indexes=indexes,
                        text=tokens_to_str(tokens, language=self.language),  
                    ),
                    label=Label(s_t)
                )
                counter_entity.update([mapper.get(s_t, s_t)])

            o_s, o_e, o_t = spo[4], spo[5], spo[6]
            tokens = self.tokens[o_s: o_e]
            indexes = list(range(o_s, o_e))
            if (o_s, o_e, o_t) not in entity_map:      
                entities[(o_s, o_e, o_t)] = Entity(     
                    span=Span(
                        tokens=tokens,
                        indexes=indexes,
                        text=tokens_to_str(tokens, language=self.language),
                    ),
                    label=Label(o_t)
                )
                counter_entity.update([mapper.get(o_t, o_t)])
            
            if spo[3] in delete_list:
                continue
            relations[spo_index] = Relation(
                arg1=entities[(s_s, s_e, s_t)],
                arg2=entities[(o_s, o_e, o_t)],
                label=Label(spo[3]),   
            )
            counter_relation.update([mapper.get(spo[3], spo[3])])

        return Sentence(
            tokens=self.tokens,
            entities=entities.values(),
            relations=relations.values(),
        ), counter_entity, counter_relation



    @staticmethod
    def load_from_file(filename, language='en', delete_list = [], m = None, logger_name='') -> List[Sentence]:
        global mapper
        mapper = m
        logger = logging.getLogger(logger_name)
        logger.info(f"Delete Relation: {delete_list}")
        
        sentence_list = list()
        raw_instance_list = json.load(open(filename))    
        logger.info(f"{filename}: {len(raw_instance_list)}")
        counter_entitys = Counter()
        counter_relations = Counter()
        count_rel = 0
        
        for instance in raw_instance_list:
            instance, counter_entity, counter_relation = JointER(
                    sentence_json=instance,
                    language=language
                ).generate_instance(delete_list)
            sentence_list += [instance]
            counter_entitys.update(counter_entity)
            counter_relations.update(counter_relation)
            if len(instance.relations) != 0:
                count_rel += 1

        counter_entitys = dict(counter_entitys)
        counter_relations = dict(counter_relations)
        counter_entitys = sorted(counter_entitys.items(), key = lambda x : x[1])
        counter_relations = sorted(counter_relations.items(), key = lambda x : x[1])
        logger.info(filename + f" Entitys: {dict(counter_entitys)}")
        logger.info(filename + f" Relations: {dict(counter_relations)}")
        logger.info(filename + f" Entitys Number: {len(counter_entitys)}")
        logger.info(filename + f" Relations Number: {len(counter_relations)}")
        logger.info(filename + f" Sentence(至少有一个relation) Number: {count_rel}")
        return sentence_list
