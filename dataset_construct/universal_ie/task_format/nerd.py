#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
from typing import List, Optional, Tuple, Set
from tqdm import tqdm
import logging
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import tokens_to_str
from universal_ie.ie_format import Entity, Label, Sentence, Span




# https://github.com/allenai/allennlp/blob/main/allennlp/data/dataset_readers/dataset_utils/span_utils.py
def _iob1_start_of_chunk(
    prev_bio_tag: Optional[str],
    prev_conll_tag: Optional[str],
    curr_bio_tag: str,
    curr_conll_tag: str,
) -> bool:
    if curr_bio_tag == "B":
        return True
    if curr_bio_tag == "I" and prev_bio_tag == "O":
        return True
    if curr_bio_tag != "O" and prev_conll_tag != curr_conll_tag:
        return True
    return False



def iob1_tags_to_spans(
    tag_sequence: List[str], classes_to_ignore: List[str] = None
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Given a sequence corresponding to IOB1 tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e., those where "B-LABEL" is not preceded
    by "I-LABEL" or "B-LABEL").
    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    prev_bio_tag = None
    prev_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        curr_bio_tag = string_tag[0]
        curr_conll_tag = string_tag[2:]

        if curr_bio_tag not in ["B", "I", "O"]:
            raise RuntimeError('Invalid tag sequence %s' % tag_sequence)
        if curr_bio_tag == "O" or curr_conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
        elif _iob1_start_of_chunk(prev_bio_tag, prev_conll_tag, curr_bio_tag, curr_conll_tag):
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = curr_conll_tag
            span_start = index
            span_end = index
        else:
            # bio_tag == "I" and curr_conll_tag == active_conll_tag
            # We're continuing a span.
            span_end += 1

        prev_bio_tag = string_tag[0]
        prev_conll_tag = string_tag[2:]
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)





class NERD(TaskFormat):
    def __init__(self, tokens: List[str], spans:  List[Tuple[Tuple[int, int], str]], language='en', instance_id=None) -> None:
        super().__init__(
            language=language
        )
        self.instance_id = instance_id
        self.tokens = tokens
        self.spans = spans

    @staticmethod
    def load_from_file(filename, language = 'en', delete_list = [], m = None, logger_name='') -> List[Sentence]:
        global mapper
        mapper = m
        logger = logging.getLogger(logger_name)
        logger.info(f"Delete Relation: {delete_list}")
        counter_entitys = Counter()
        count_entitys = 0

        sentence_list = list()
        for rows in tqdm(NERD.generate_sentence(filename)):
            if rows[0][0] == '-DOCSTART-':
                continue
            tokens = [token[0] for token in rows]
            ner = [token[1] for token in rows]
            spans = iob1_tags_to_spans(ner)
            spans = [
                {'start': span[1][0], 'end': span[1][1], 'type': span[0]}
                for span in spans
            ]
            sentence, counter_entity = NERD(
                tokens=tokens,
                spans=spans,
                language=language,
            ).generate_instance(delete_list)

            counter_entitys.update(counter_entity)
            sentence_list += [sentence]
            if len(sentence.entities) != 0:
                count_entitys += 1

        counter_entitys = sorted(counter_entitys.items(), key = lambda x : x[1])
        logger.info(filename + f" Entitys: {dict(counter_entitys)}")
        logger.info(filename + f" Entitys Number: {len(counter_entitys)}")
        logger.info(filename + f" Sentence(至少有一个entity) Number: {count_entitys}")
        return sentence_list


    @staticmethod
    def generate_sentence(filename):
        sentence = list()
        with open(filename) as fin:
            for line in fin:
                if line.strip() == '':
                    if len(sentence) != 0:
                        yield sentence
                        sentence = list()

                else:
                    sentence += [line.strip().split()]       

            if len(sentence) != 0:
                yield sentence


    def generate_instance(self, delete_list):
        counter_entity = Counter()
        entities = list()
        for span_index, span in enumerate(self.spans):
            tokens = self.tokens[span['start']: span['end'] + 1]
            indexes = list(range(span['start'], span['end'] + 1))
            if span['type'] in delete_list:
                continue
            entities += [
                Entity(
                    span=Span(
                        tokens=tokens,
                        indexes=indexes,
                        text=tokens_to_str(tokens, language=self.language),
                        text_id=self.instance_id
                    ),
                    label=Label(span['type']),
                    text_id=self.instance_id,
                    record_id=self.instance_id + "#%s" % span_index if self.instance_id else None)
            ]
            counter_entity.update([mapper.get(span['type'], span['type'])])

        return Sentence(tokens=self.tokens,
                        entities=entities,
                        text_id=self.instance_id), counter_entity



