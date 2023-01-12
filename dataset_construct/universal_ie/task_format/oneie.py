#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
from typing import Counter, List
import logging

from universal_ie.utils import tokens_to_str
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.ie_format import Entity, Event, Label, Sentence, Span



class OneIEEvent(TaskFormat):
    def __init__(self, doc_json, language='en'):
        super().__init__(
            language=language
        )
        self.doc_id = doc_json['doc_id']
        self.sent_id = doc_json['sent_id']
        self.tokens = doc_json['tokens']
        self.entities = doc_json['entity_mentions']
        self.relations = doc_json['relation_mentions']
        self.events = doc_json['event_mentions']

    def generate_instance(self, delete_list):
        events = dict()
        entities = dict()
        counter_trigger = Counter()
        counter_role = Counter()

        for span_index, span in enumerate(self.entities):
            tokens = self.tokens[span['start']: span['end']]
            indexes = list(range(span['start'], span['end']))
            entities[span['id']] = Entity(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id
                ),
                label=Label(span['entity_type']),
                text_id=self.sent_id,
                record_id=span['id']
            )

        for event_index, event in enumerate(self.events):
            if str(event['event_type']) in delete_list:
              continue
            start = event['trigger']['start']
            end = event['trigger']['end']
            tokens = self.tokens[start:end]
            indexes = list(range(start, end))
            events[event['id']] = Event(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id
                ),
                label=Label(event['event_type']),
                args=[(Label(x['role']), entities[x['entity_id']])
                      for x in event['arguments']],
                text_id=self.sent_id,
                record_id=event['id']
            )
            counter_trigger.update([mapper.get(event['event_type'], event['event_type'])])
            for x in event['arguments']:
              counter_role.update([mapper.get(x['role'], x['role'])])

        return Sentence(
            tokens=self.tokens,
            entities=list(),
            relations=list(),
            events=events.values(),
            text_id=self.sent_id
        ), counter_trigger, counter_role

    @staticmethod
    def load_from_file(filename, language='en', delete_list = [], m = None, logger_name='') -> List[Sentence]:
        global mapper
        mapper = m
        logger = logging.getLogger(logger_name)
        logger.info(f"Delete Trigger: {delete_list}")
        sentence_list = list()
        counter_triggers = Counter()
        counter_roles = Counter()
        count_sent = 0
        with open(filename) as fin:
            for line in fin:
                instance, counter_trigger, counter_role = OneIEEvent(
                    json.loads(line.strip()),
                    language=language
                ).generate_instance(delete_list)
                sentence_list += [instance]
                counter_triggers.update(counter_trigger)
                counter_roles.update(counter_role)
                if len(instance.events) != 0:
                  count_sent += 1

        counter_triggers = sorted(dict(counter_triggers).items(), key = lambda x : x[1])
        counter_roles = sorted(dict(counter_roles).items(), key = lambda x : x[1])
        logger.info(filename + f" Event Trigger: {dict(counter_triggers)}")
        logger.info(filename + f" Event Role: {dict(counter_roles)}")
        logger.info(filename + f" Event Trigger Number: {len(counter_triggers)}")
        logger.info(filename + f" Event Role Number: {len(counter_roles)}")
        logger.info(filename + f" Sentence(至少有一个event) Number: {count_sent}")
        return sentence_list
        
