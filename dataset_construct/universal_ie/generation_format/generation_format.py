#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Dict, Union
from collections import defaultdict
from universal_ie.record_schema import RecordSchema
from universal_ie.generation_format.structure_marker import StructureMarker
from universal_ie.ie_format import Entity, Relation, Event, Label
import abc


class GenerationFormat:
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 structure_maker: StructureMarker,
                 label_mapper: Dict = None,
                 language: str = 'en') -> None:
        self.structure_maker = structure_maker
        self.language = language
        self.label_mapper = {} if label_mapper is None else label_mapper
        self.record_role_map = defaultdict(set) 

    def get_label_str(self, label: Label):
        return self.label_mapper.get(label.__repr__(), label.__repr__())

    @abc.abstractmethod
    def annotate_entities(
        self, tokens: List[str], entities: List[Entity]): pass

    @abc.abstractmethod
    def annotate_given_entities(self, tokens: List[str], entities: Union[List[Entity], Entity]): pass

    @abc.abstractmethod
    def annotate_events(self, tokens: List[str], events: List[Event]): pass

    @abc.abstractmethod
    def annotate_event_given_predicate(self, tokens: List[str], event: Event): pass

    @abc.abstractmethod
    def annotate_relation_extraction(self, tokens: List[str],
                                     relations: List[Relation]): pass

    def output_schema(self, filename: str):
        """导出 Schema 文件
        Args:
            filename (str): [description]
        """
        record_list = list(self.record_role_map.keys())
        role_set = set()
        for record in self.record_role_map:
            role_set.update(self.record_role_map[record])
            self.record_role_map[record] = list(self.record_role_map[record])
        role_list = list(role_set)

        record_schema = RecordSchema(type_list=record_list,
                                     role_list=role_list,
                                     type_role_dict=self.record_role_map)
        record_schema.write_to_file(filename)
