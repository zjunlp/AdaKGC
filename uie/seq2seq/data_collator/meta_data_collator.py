#!/usr/bin/env python
# -*- coding:utf-8 -*-
from itertools import count
import os
from dataclasses import dataclass
import torch
import logging
import random
import math
from typing import Optional, Union
from collections import OrderedDict
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers.file_utils import PaddingStrategy

from uie.extraction.record_schema import RecordSchema
from uie.extraction.constants import BaseStructureMarker, text_start, type_start, type_end, spot_prompt, asoc_prompt, span_start, null_span
from uie.extraction.utils import convert_to_record_function
from uie.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser


logger = logging.getLogger("__main__")


class DynamicSSIGenerator():
    """
    Sample negative spot and asoc to construct SSI, meta schema需要添加一些negative spot和asoc
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, schema: RecordSchema, negative_list=[], positive_rate=1, spot_negative=5, asoc_negative=5, other_ratio=0.3, ordered_prompt=False, task_name=None) -> None:
        self.spot_dict = self.get_ordered_dict(schema.type_list, tokenizer)    # {'aspect':[2663], ...}
        self.asoc_dict = self.get_ordered_dict(schema.role_list, tokenizer)    # {'positive':[1465], ...}
        self.spot_list = list(self.spot_dict.keys())
        self.asoc_list = list(self.asoc_dict.keys())
        self.spot_prompt = tokenizer.get_vocab()[spot_prompt]    # spot_prompt 是 '<spot>'，从constants里导入的
        self.asoc_prompt = tokenizer.get_vocab()[asoc_prompt]
        self.text_start = tokenizer.get_vocab()[text_start]
        self.positive_rate = positive_rate if positive_rate > 0 and positive_rate < 1 else 1
        self.spot_negative = spot_negative
        self.asoc_negative = asoc_negative
        self.ordered_prompt = ordered_prompt
        self.task_name = task_name

    @staticmethod
    def get_ordered_dict(schema_name_list, tokenizer):
        schema_ordered_dict = OrderedDict()
        for name in schema_name_list:
            schema_ordered_dict[name] = tokenizer.encode(name, add_special_tokens=False)
        return schema_ordered_dict

    @staticmethod
    def sample_negative(postive, candidates, k=5):
        if k < 0:
            k = len(candidates)
        negative_set = set()
        for index in torch.randperm(len(candidates))[:k].tolist():
            negative = candidates[index]
            if negative not in postive:
                negative_set.add(negative)
        return list(negative_set)

    def sample_spot(self, positive):
        """ Sample spot
        """
        if self.task_name == 'relation':
            negative_spot = self.sample_negative(postive=positive, candidates=self.spot_list, k=self.spot_negative)
        else:
            negative_spot = self.sample_negative(postive=positive, candidates=self.spot_list, k=random.randint(self.spot_negative,len(self.spot_list)))
        prefix_spot_candidates = positive + negative_spot

        converted_spot_prefix = self.convert_prefix(
            candidates=prefix_spot_candidates,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=self.ordered_prompt,
        )

        return converted_spot_prefix, positive, negative_spot

    def sample_asoc(self, positive):
        """ Sample Asoc
        """
        if self.task_name == 'relation':
            negative_asoc = self.sample_negative(postive=positive, candidates=self.asoc_list, k=random.randint(self.asoc_negative,len(self.asoc_list)))
        else:
            negative_asoc = self.sample_negative(postive=positive, candidates=self.asoc_list, k=self.asoc_negative)
        prefix_asoc_candidates = positive + negative_asoc
        converted_asoc_prefix = self.convert_prefix(
            candidates=prefix_asoc_candidates,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=self.ordered_prompt,
        )
        return converted_asoc_prefix, positive, negative_asoc

    def full_spot(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.spot_list,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=ordered_prompt,
        )

    def full_asoc(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.asoc_list,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=ordered_prompt,
        )

    def full_null(self, negative_spot, negative_asoc):
        full_null_text = []
        if self.task_name == 'relation':
            for asoc in negative_asoc:
                asoc_str_rep = ' '.join([
                    type_start,
                    asoc,
                    span_start,
                    null_span,
                    type_end,
                ])
                full_null_text.append(asoc_str_rep)
        else:
            for spot in negative_spot:
                spot_str_rep = ' '.join([
                    type_start,
                    spot,
                    span_start,
                    null_span,
                    type_end,
                ])
                full_null_text.append(spot_str_rep)    
        return ' '.join(full_null_text)





    @staticmethod
    def convert_prefix(candidates, prompt, mapper, ordered_prompt=True):
        prefix = list()
        if ordered_prompt:
            candidate_sorted = sorted([(candidate, index) for index, candidate in enumerate(candidates)])
            index_list = [index for _, index in candidate_sorted]
        else:
            index_list = torch.randperm(len(candidates)).tolist()

        for index in index_list:
            prefix += [prompt]
            prefix += mapper[candidates[index]]
        return prefix




@dataclass
class DataCollatorForMetaSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    negative_sampler: DynamicSSIGenerator
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    max_target_length: Optional[int] = None
    max_prefix_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    spot_asoc_nosier: SpotAsocNoiser = None
    decoding_format: str = 'spotasoc'
    add_null: bool = False
    use_ssi: bool = True
    count : int = 10

    def __call__(self, features):
        """ Make Meta Schema Batch
        Args:
            features (Dict): [description]
                - sample_prompt: indicates sample_prompt example, need pop after call
                - spots (List[str]): List of spots in this sentence, need pop after call
                - asocs (List[str]): List of asocs in this sentence, need pop after call
                - input_ids
                - attention_mask
                - labels

        Returns:
        """
        for feature in features:

            sample_prompt = feature['sample_prompt']
            if self.count > 0:
                logger.info(f"{self.count}/10")
                logger.info(f"feature['input_ids']: {self.tokenizer.convert_ids_to_tokens(feature['input_ids'])}")   # feature['input_ids']: Parts of the Pennsylvania Turnpike were closed ; so too was the Baseball Hall of Fame and Museum in Cooperstown, N.Y. Then there was the house that was spotted drifting down the Susquehanna River in New York -- on fire for a while, it seemed. "</s>
                logger.info(f"feature['labels']: {self.tokenizer.convert_ids_to_tokens(feature['labels'])}")
                logger.info(f"feature['input_ids']: {feature['input_ids']}")   # feature['input_ids']: Parts of the Pennsylvania Turnpike were closed ; so too was the Baseball Hall of Fame and Museum in Cooperstown, N.Y. Then there was the house that was spotted drifting down the Susquehanna River in New York -- on fire for a while, it seemed. "</s>
                logger.info(f"feature['labels']: {feature['labels']}")         # feature['labels']: <extra_id_0><extra_id_0> location<extra_id_5> Cooperstown<extra_id_1><extra_id_0> location<extra_id_5> Susquehanna River<extra_id_1><extra_id_0> location<extra_id_5> New York<extra_id_0> contains<extra_id_5> Cooperstown<extra_id_1><extra_id_0> contains<extra_id_5> Susquehanna River<extra_id_1><extra_id_1><extra_id_1></s>

            if not sample_prompt:
                # Evaluation using Ordered SSI
                converted_spot_prefix = self.negative_sampler.full_spot(shuffle=self.model.training)
                converted_asoc_prefix = self.negative_sampler.full_asoc(shuffle=self.model.training)
            else:
                # Sample SSI，采样negtive shema
                converted_spot_prefix, positive_spot, negative_spot = self.negative_sampler.sample_spot(positive=feature.get('spots', []))
                converted_asoc_prefix, positive_asoc, negative_asoc = self.negative_sampler.sample_asoc(positive=feature.get('asocs', []))
                # 传入的 positive=feature.get('asocs'/'spots')是该feature的真实label
                if self.count > 0:
                    logger.info(f"Converted_Spot_Prefix: {self.tokenizer.decode(converted_spot_prefix)}") 
                    logger.info(f"Converted_Asoc_Prefix: {self.tokenizer.decode(converted_asoc_prefix)}") 
                    logger.info(f"Positive_Spot Len: {len(positive_spot)} \t {positive_spot}")  
                    logger.info(f"Positive_Asoc Len: {len(positive_asoc)} \t {positive_asoc}")
                    logger.info(f"Negative_Spot Len: {len(negative_spot)} \t {negative_spot}")  
                    logger.info(f"Negative_Asoc Len: {len(negative_asoc)} \t {negative_asoc}")
                # Dynamic generating spot-asoc during training，evaluating时也有
                if 'spot_asoc' in feature:
                    # Deleted positive example Spot in Target that was not sampled by Prefix
                    feature['spot_asoc'] = [spot_asoc for spot_asoc in feature['spot_asoc'] if spot_asoc["label"] in positive_spot]
                    if self.add_null:
                        null_text = self.negative_sampler.full_null(negative_spot, negative_asoc)
                        record = convert_to_record_function[self.decoding_format](
                            spot_asoc_instance = feature["spot_asoc"],
                            structure_maker = BaseStructureMarker(),
                            null_text = null_text,
                        )
                    else:
                        feature['spot_asoc'] = self.spot_asoc_nosier.add_noise(
                            feature['spot_asoc'],
                            spot_label_list=negative_spot,
                            asoc_label_list=negative_asoc,
                        )
                        record = convert_to_record_function[self.decoding_format](
                            spot_asoc_instance = feature["spot_asoc"],
                            structure_maker = BaseStructureMarker(),
                        )
                    feature["labels"] = self.tokenizer.encode(record)

                    if self.count > 0:
                        logger.info(f"Record: {record}")   # Record: <extra_id_0> <extra_id_0> organization <extra_id_5> <extra_id_6> <extra_id_1> <extra_id_0> location <extra_id_5> Cooperstown <extra_id_1> <extra_id_0> location <extra_id_5> Susquehanna River <extra_id_1> <extra_id_0> location <extra_id_5> New York <extra_id_0> contains <extra_id_5> Cooperstown <extra_id_1> <extra_id_0> contains <extra_id_5> Susquehanna River <extra_id_1> <extra_id_1> <extra_id_1>
                        logger.info(f"feature['labels']: {self.tokenizer.convert_ids_to_tokens(feature['labels'])}")   # 同Record不过是encode后的
                        logger.info(f"feature['labels']: {feature['labels']}")   # 同Record不过是encode后的


            feature.pop('sample_prompt') if 'sample_prompt' in feature else None
            feature.pop('spot_asoc') if 'spot_asoc' in feature else None
            feature.pop('spots') if 'spots' in feature else None
            feature.pop('asocs') if 'asocs' in feature else None

            prefix = converted_spot_prefix + converted_asoc_prefix
            # truncate `prefix` to max length
            if self.max_prefix_length is not None and self.max_prefix_length >= 0:
                prefix = prefix[:self.max_prefix_length]

            feature['input_ids'] = prefix + [self.negative_sampler.text_start] + feature['input_ids']  # <text>分隔符
            # truncate `input_ids` to max length

            if self.count > 0:
                logger.info(f"Prefix: {self.tokenizer.convert_ids_to_tokens(prefix)}")     # <spot> organization<spot> location<spot> person<asoc> major shareholder of<asoc> people<asoc> profession<asoc> children<asoc> place of death<asoc> advisors<asoc> teams<asoc> contains<asoc> industry<asoc> place founded
                logger.info(f"feature['input_ids']: {self.tokenizer.convert_ids_to_tokens(feature['input_ids'])}")    # feature['input_ids']：<spot> organization<spot> location<spot> person<asoc> major shareholder of<asoc> people<asoc> profession<asoc> children<asoc> place of death<asoc> advisors<asoc> teams<asoc> contains<asoc> industry<asoc> place founded<extra_id_2> Parts of the Pennsylvania Turnpike were closed ; so too was the Baseball Hall of Fame and Museum in Cooperstown, N.Y. Then there was the house that was spotted drifting down the Susquehanna River in New York -- on fire for a while, it seemed. "</s>
                logger.info(f"Prefix: {prefix}")     # <spot> organization<spot> location<spot> person<asoc> major shareholder of<asoc> people<asoc> profession<asoc> children<asoc> place of death<asoc> advisors<asoc> teams<asoc> contains<asoc> industry<asoc> place founded
                logger.info(f"feature['input_ids']: {feature['input_ids']}")    # feature['input_ids']：<spot> organization<spot> location<spot> person<asoc> major shareholder of<asoc> people<asoc> profession<asoc> children<asoc> place of death<asoc> advisors<asoc> teams<asoc> contains<asoc> industry<asoc> place founded<extra_id_2> Parts of the Pennsylvania Turnpike were closed ; so too was the Baseball Hall of Fame and Museum in Cooperstown, N.Y. Then there was the house that was spotted drifting down the Susquehanna River in New York -- on fire for a while, it seemed. "</s>
                # Prefix取了所有label，'input_ids'在这里添加前缀

            if self.max_length:
                feature['input_ids'] = feature['input_ids'][:self.max_length]
            if self.max_target_length and 'labels' in feature:
                feature['labels'] = feature['labels'][:self.max_target_length]

            feature['attention_mask'] = [1] * len(feature['input_ids'])

            if self.count > 0:
                self.count -= 1

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        if labels is not None:
            max_label_length = max(len(_label) for _label in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        
        return features




class PromptSSIGenerator():
    def __init__(self, tokenizer, schema: RecordSchema, negative_list=[], positive_rate=1, spot_negative=5, asoc_negative=5, ordered_prompt=False, other_ratio=0.3, task_name='') -> None:
        self.spot_dict = self.get_ordered_dict(schema.type_list, tokenizer)    # {'aspect':[2663], ...}
        self.asoc_dict = self.get_ordered_dict(schema.role_list, tokenizer)    # {'positive':[1465], ...}
        self.spot_list = list(self.spot_dict.keys())
        self.asoc_list = list(self.asoc_dict.keys())
        self.spot_prompt = tokenizer.get_vocab()[spot_prompt]    
        self.asoc_prompt = tokenizer.get_vocab()[asoc_prompt]
        self.text_start = tokenizer.get_vocab()[text_start]
        self.negative_list = negative_list
        self.other_rate = other_ratio
        self.positive_rate = positive_rate if positive_rate > 0 and positive_rate < 1 else 1
        self.spot_negative = spot_negative
        self.asoc_negative = asoc_negative
        self.ordered_prompt = ordered_prompt
        self.task_name = task_name

    @staticmethod
    def get_ordered_dict(schema_name_list, tokenizer):
        schema_ordered_dict = OrderedDict()
        for name in schema_name_list:
            schema_ordered_dict[name] = tokenizer.encode(name, add_special_tokens=False)
        return schema_ordered_dict


    @staticmethod
    def sample_negative(postive, candidates, k = 5):
        if k < 0:
            k = len(candidates)
        negative_set = set()
        for index in torch.randperm(len(candidates))[: k].tolist():
            negative = candidates[index]
            if negative not in postive:
                negative_set.add(negative)
        return list(negative_set)


    def sample_spot(self, positive):
        if self.task_name != 'relation':
            negative_spot = self.sample_negative(postive = positive, candidates = self.spot_list + self.negative_list[: int(self.other_rate * len(self.spot_list))], k = self.spot_negative)
        else:
            negative_spot = self.sample_negative(postive = positive, candidates = self.spot_list, k = self.spot_negative)
        prefix_spot_candidates = positive + negative_spot

        converted_spot_prefix = self.convert_prefix(
            candidates=prefix_spot_candidates,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=self.ordered_prompt,
        )
        if self.ordered_prompt:
            prefix_spot_candidates = sorted(prefix_spot_candidates)

        return prefix_spot_candidates, converted_spot_prefix, positive, negative_spot


    def sample_asoc(self, positive):
        if self.task_name == 'relation':
            negative_asoc = self.sample_negative(postive = positive, candidates = self.asoc_list + self.negative_list[: int(self.other_rate * len(self.asoc_list))], k = self.asoc_negative)
        else:
            negative_asoc = self.sample_negative(postive = positive, candidates = self.asoc_list, k = self.asoc_negative)
        prefix_asoc_candidates = positive + negative_asoc

        converted_asoc_prefix = self.convert_prefix(
            candidates=prefix_asoc_candidates,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=self.ordered_prompt,
        )
        if self.ordered_prompt:
            prefix_asoc_candidates = sorted(prefix_asoc_candidates)

        return prefix_asoc_candidates, converted_asoc_prefix, positive, negative_asoc


    def full_spot(self, shuffle=False):
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True

        spot_list = self.spot_list
        if ordered_prompt:
            spot_list = sorted(self.spot_list)

        convert_spot = self.convert_prefix(
            candidates=self.spot_list,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=ordered_prompt,
        )
            
        return spot_list, convert_spot


    def full_asoc(self, shuffle=False):
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True

        asoc_list = self.asoc_list
        if ordered_prompt:
            asoc_list = sorted(self.asoc_list)

        convert_asoc = self.convert_prefix(
            candidates=self.asoc_list,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=ordered_prompt,
        )

        return asoc_list, convert_asoc


    def full_null(self, negative_spot, negative_asoc):
        full_null_text = []
        if self.task_name == 'relation':
            for asoc in negative_asoc:
                asoc_str_rep = ' '.join([
                    type_start,
                    asoc,
                    span_start,
                    null_span,
                    type_end,
                ])
                full_null_text.append(asoc_str_rep)
        else:
            for spot in negative_spot:
                spot_str_rep = ' '.join([
                    type_start,
                    spot,
                    span_start,
                    null_span,
                    type_end,
                ])
                full_null_text.append(spot_str_rep)    
        return ' '.join(full_null_text)


    @staticmethod
    def convert_prefix(candidates, prompt, mapper, ordered_prompt=True):
        prefix = list()
        if ordered_prompt:
            candidate_sorted = sorted([(candidate, index) for index, candidate in enumerate(candidates)])
            index_list = [index for _, index in candidate_sorted]
        else:
            index_list = torch.randperm(len(candidates)).tolist()

        for index in index_list:
            prefix += [prompt]
            prefix += mapper[candidates[index]]
        return prefix



@dataclass
class PromptForMetaSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    negative_sampler: PromptSSIGenerator
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    max_target_length: Optional[int] = None
    max_prefix_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    spot_asoc_nosier: SpotAsocNoiser = None
    decoding_format: str = 'spotasoc'
    use_ssi: bool = True
    add_null: bool = False
    count : int = 10

    def __call__(self, features):
        for feature in features:
            sample_prompt = feature['sample_prompt']
            if self.count > 0:
                logger.info(f"{self.count}/10")
                logger.info(f"feature['input_ids']: {self.tokenizer.convert_ids_to_tokens(feature['input_ids'])}")   # feature['input_ids']: Parts of the Pennsylvania Turnpike were closed ; so too was the Baseball Hall of Fame and Museum in Cooperstown, N.Y. Then there was the house that was spotted drifting down the Susquehanna River in New York -- on fire for a while, it seemed. "</s>
                logger.info(f"feature['labels']: {self.tokenizer.convert_ids_to_tokens(feature['labels'])}")         # feature['labels']: <extra_id_0><extra_id_0> location<extra_id_5> Cooperstown<extra_id_1><extra_id_0> location<extra_id_5> Susquehanna River<extra_id_1><extra_id_0> location<extra_id_5> New York<extra_id_0> contains<extra_id_5> Cooperstown<extra_id_1><extra_id_0> contains<extra_id_5> Susquehanna River<extra_id_1><extra_id_1><extra_id_1></s>

            '''评估(test)所有的spot、asoc都要用到, 不存在正负样本'''
            if not sample_prompt:
                spot_prefix, convert_spot = self.negative_sampler.full_spot(shuffle=self.model.training)
                asoc_prefix, convert_asoc = self.negative_sampler.full_asoc(shuffle=self.model.training)
            else:
                '''
                获得正样本(positive_spot)和负样本(negative_spot), 
                spot_prefix是正spot+负spot,
                convert_spot是在每个spot之间添加了特殊分隔符(<spot>)
                '''
                spot_prefix, convert_spot, positive_spot, negative_spot = self.negative_sampler.sample_spot(positive=feature.get('spots', []))
                asoc_prefix, convert_asoc, positive_asoc, negative_asoc = self.negative_sampler.sample_asoc(positive=feature.get('asocs', []))
                if self.count > 0:
                    logger.info(f"Spot_Prefix: {spot_prefix}") 
                    logger.info(f"Asoc_Prefix: {asoc_prefix}") 
                    logger.info(f"Positive_Spot Len: {len(positive_spot)} \t {positive_spot}")  
                    logger.info(f"Positive_Asoc Len: {len(positive_asoc)} \t {positive_asoc}")
                    logger.info(f"Negative_Spot Len: {len(negative_spot)} \t {negative_spot}")  
                    logger.info(f"Negative_Asoc Len: {len(negative_asoc)} \t {negative_asoc}")
                '''
                feature['input_ids']: ['▁So', '▁to', '▁me', '▁', ',', '▁the', '▁key', '▁thing', '▁is', '▁that', '▁we', '▁ought', '▁to', '▁be', '▁taking', '▁care', '▁of', '▁the', '▁military', '▁and', '▁that', '▁', "'", '▁', 's', '▁what', '▁we', '▁should', '▁do', '▁', '.', '</s>']
                feature['labels']: ['<extra_id_0>', '<extra_id_1>', '</s>']
                Positive_Spot Len: 0 	 []
                Positive_Asoc Len: 0 	 []
                Negative_Spot Len: 12 	 ['sentence', 'marry', 'phone write', 'sue', 'be born', 'acquit', 'elect', 'transfer ownership', 'attack', 'extradite', 'end organization', 'arrest jail']
                Negative_Asoc Len: 19 	 ['adjudicator', 'beneficiary', 'place', 'instrument', 'destination', 'organization', 'buyer', 'plaintiff', 'defendant', 'target', 'victim', 'artifact', 'person', 'attacker', 'entity', 'seller', 'agent', 'origin', 'vehicle']
                上面的数据中可以看到label中的正样本数是0, 负采样后得到12、19个负样本
                '''

                # Dynamic generating spot-asoc during training，evaluating时也有
                if 'spot_asoc' in feature:
                    feature['spot_asoc'] = [spot_asoc for spot_asoc in feature['spot_asoc'] if spot_asoc["label"] in positive_spot]
                    if self.add_null:
                        null_text = self.negative_sampler.full_null(negative_spot, negative_asoc)
                        record = convert_to_record_function[self.decoding_format](
                            spot_asoc_instance = feature["spot_asoc"],
                            structure_maker = BaseStructureMarker(),
                            null_text = null_text,
                        )
                    else:
                        feature['spot_asoc'] = self.spot_asoc_nosier.add_noise(
                            feature['spot_asoc'],
                            spot_label_list=negative_spot,
                            asoc_label_list=negative_asoc,
                        )
                        record = convert_to_record_function[self.decoding_format](
                            spot_asoc_instance = feature["spot_asoc"],
                            structure_maker = BaseStructureMarker(),
                        )
                    feature["labels"] = self.tokenizer.encode(record)

                    if self.count > 0:
                        logger.info(f"Record: {record}")   
                        logger.info(f"feature['labels']: {self.tokenizer.convert_ids_to_tokens(feature['labels'])}")  


            feature.pop('sample_prompt') if 'sample_prompt' in feature else None
            feature.pop('spot_asoc') if 'spot_asoc' in feature else None
            '''
            spot、asoc由原先的正样本变为负采样后的正加负
            "spot": ["geographical social political"], 
            采样两个负样本"person other", "writtenart"
            最终得到"spot": ["geographical social political", "person other", "writtenart"], 
            feature['spot']、feature['asoc']在获得prompt的时候用到
            '''
            feature['spot'] = [self.tokenizer.encode(s, add_special_tokens = False) for s in spot_prefix]
            feature['asoc'] = [self.tokenizer.encode(a, add_special_tokens = False) for a in asoc_prefix]

            if self.use_ssi:
                prefix = convert_spot + convert_asoc
                if self.max_prefix_length is not None and self.max_prefix_length >= 0:
                    prefix = prefix[:self.max_prefix_length]
                feature['input_ids'] = prefix + [self.negative_sampler.text_start] + feature['input_ids']  # <text>分隔符
                if self.count > 0:
                    logger.info(f"Prefix: {self.tokenizer.convert_ids_to_tokens(prefix)}")     # <spot> organization<spot> location<spot> person<asoc> major shareholder of<asoc> people<asoc> profession<asoc> children<asoc> place of death<asoc> advisors<asoc> teams<asoc> contains<asoc> industry<asoc> place founded
                    logger.info(f"feature['input_ids']: {self.tokenizer.convert_ids_to_tokens(feature['input_ids'])}") 
                    '''
                    Prefix: ['<spot>', '▁', 'a', 'c', 'quit', '<spot>', '▁arrest', '▁jail', '<spot>', '▁attack', '<spot>', '▁be', '▁born', '<spot>', '▁elect', '<spot>', '▁end', '▁organization', '<spot>', '▁extra', 'dite', '<spot>', '▁marry', '<spot>', '▁phone', '▁write', '<spot>', '▁sentence', '<spot>', '▁su', 'e', '<spot>', '▁transfer', '▁ownership', '<asoc>', '▁adj', 'u', 'dic', 'ator', '<asoc>', '▁agent', '<asoc>', '▁art', 'i', 'fact', '<asoc>', '▁attacker', '<asoc>', '▁beneficiary', '<asoc>', '▁buyer', '<asoc>', '▁defendant', '<asoc>', '▁destination', '<asoc>', '▁entity', '<asoc>', '▁instrument', '<asoc>', '▁organization', '<asoc>', '▁origin', '<asoc>', '▁person', '<asoc>', '▁place', '<asoc>', '▁plaintiff', '<asoc>', '▁seller', '<asoc>', '▁target', '<asoc>', '▁vehicle', '<asoc>', '▁victim']
                    feature['input_ids']: ['<spot>', '▁', 'a', 'c', 'quit', '<spot>', '▁arrest', '▁jail', '<spot>', '▁attack', '<spot>', '▁be', '▁born', '<spot>', '▁elect', '<spot>', '▁end', '▁organization', '<spot>', '▁extra', 'dite', '<spot>', '▁marry', '<spot>', '▁phone', '▁write', '<spot>', '▁sentence', '<spot>', '▁su', 'e', '<spot>', '▁transfer', '▁ownership', '<asoc>', '▁adj', 'u', 'dic', 'ator', '<asoc>', '▁agent', '<asoc>', '▁art', 'i', 'fact', '<asoc>', '▁attacker', '<asoc>', '▁beneficiary', '<asoc>', '▁buyer', '<asoc>', '▁defendant', '<asoc>', '▁destination', '<asoc>', '▁entity', '<asoc>', '▁instrument', '<asoc>', '▁organization', '<asoc>', '▁origin', '<asoc>', '▁person', '<asoc>', '▁place', '<asoc>', '▁plaintiff', '<asoc>', '▁seller', '<asoc>', '▁target', '<asoc>', '▁vehicle', '<asoc>', '▁victim', '<extra_id_2>', '▁So', '▁to', '▁me', '▁', ',', '▁the', '▁key', '▁thing', '▁is', '▁that', '▁we', '▁ought', '▁to', '▁be', '▁taking', '▁care', '▁of', '▁the', '▁military', '▁and', '▁that', '▁', "'", '▁', 's', '▁what', '▁we', '▁should', '▁do', '▁', '.', '</s>']
                    Prefix是spot、asoc之间添加了特殊分隔符(<spot>、<asoc>)的格式
                    如果use_ssi==True, 会在模型的输入'input_ids'的前面添加prefix
                    '''

            if self.max_length:
                feature['input_ids'] = feature['input_ids'][:self.max_length]
            if self.max_target_length and 'labels' in feature:
                feature['labels'] = feature['labels'][:self.max_target_length]
            
            feature['attention_mask'] = [1] * len(feature['input_ids'])
            if self.max_length:
                feature['input_ids'] = feature['input_ids'] + [self.tokenizer.pad_token_id] * (self.max_length - len(feature['input_ids']))
                feature['attention_mask'] = feature['attention_mask'] + [0] * (self.max_length - len(feature['attention_mask']))

            if self.count > 0:
                self.count -= 1


        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        if labels is not None:
            max_label_length = max(len(_label) for _label in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
                feature['labels'] = torch.tensor(feature['labels'], dtype = torch.long).unsqueeze(0)

        for feature in features:
            feature['input_ids'] = torch.tensor(feature['input_ids'], dtype = torch.long).unsqueeze(0)
            feature['attention_mask'] = torch.tensor(feature['attention_mask'], dtype = torch.long).unsqueeze(0)

        max_spot_len = max(len(feature["spot"]) for feature in features)
        max_asoc_len = max(len(feature["asoc"]) for feature in features)

        for feature in features:
            feature['spot'] = feature['spot'] + [(max_spot_len - len(feature['spot'])) * [0]]
            feature['asoc'] = feature['asoc'] + [(max_asoc_len - len(feature['asoc'])) * [0]]

        examples = {}
        for feature in features:
            if 'input_ids' not in examples.keys():
                examples['input_ids'] = feature['input_ids']
                examples['attention_mask'] = feature['attention_mask']
                if 'labels' in feature.keys():
                    examples['labels'] = feature['labels']
                examples['spot'] = [feature['spot'], ]
                examples['asoc'] = [feature['asoc'], ]
            else:
                examples['input_ids'] = torch.cat([examples['input_ids'], feature['input_ids']], dim=0)
                examples['attention_mask'] = torch.cat([examples['attention_mask'], feature['attention_mask']], dim=0)
                if 'labels' in feature.keys():
                    examples['labels'] = torch.cat([examples['labels'], feature['labels']], dim=0)
                examples['spot'].append(feature['spot'])
                examples['asoc'].append(feature['asoc'])

        # prepare decoder_input_idsf
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            examples["decoder_input_ids"] = decoder_input_ids
        
        return examples        # 返回的examples应该是字典dict{tuple()}, features是字典元组tuple(dict{})

        