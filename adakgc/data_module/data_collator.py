#!/usr/bin/env python
# -*- coding:utf-8 -*-
from dataclasses import dataclass
import torch
from typing import Optional, Union
from collections import OrderedDict
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers.file_utils import PaddingStrategy

from adakgc.data_module.spot_asoc_noiser import SpotAsocNoiser
from adakgc.utils.record_schema import RecordSchema
from adakgc.utils.constants import BaseStructureMarker, text_start, type_start, type_end, spot_prompt, asoc_prompt, span_start, null_span
from adakgc.utils.utils import convert_to_record_function




class PromptSSIGenerator():
    def __init__(self, tokenizer, schema: RecordSchema, negative_list=[], positive_rate=1, spot_negative=5, asoc_negative=5, ordered_prompt=False, other_ratio = 0.3) -> None:
        self.spot_dict = self.get_ordered_dict(schema.type_list, tokenizer)   
        self.asoc_dict = self.get_ordered_dict(schema.role_list, tokenizer)    
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
        negative_spot = self.sample_negative(postive = positive, candidates = self.spot_list + self.negative_list[: int(self.other_rate * len(self.spot_list))], k = self.spot_negative)
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





class RelationPromptSSIGenerator(PromptSSIGenerator):
    def __init__(self, tokenizer, schema: RecordSchema, negative_list=[], positive_rate=1, spot_negative=5, asoc_negative=5, ordered_prompt=False, other_ratio = 0.3) -> None:
        super.__init__(tokenizer, schema, negative_list, positive_rate, spot_negative, asoc_negative, ordered_prompt, other_ratio)


    def sample_spot(self, positive):
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
        negative_asoc = self.sample_negative(postive = positive, candidates = self.asoc_list + self.negative_list[: int(self.other_rate * len(self.asoc_list))], k = self.asoc_negative)
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






@dataclass
class PromptDataCollatorForMetaSeq2Seq:
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

    def __call__(self, features):
        for feature in features:
            sample_prompt = feature['sample_prompt']

            if not sample_prompt:
                # Evaluation using Ordered SSI
                spot_prefix, convert_spot = self.negative_sampler.full_spot(shuffle=self.model.training)
                asoc_prefix, convert_asoc = self.negative_sampler.full_asoc(shuffle=self.model.training)
            else:
                # Sample SSI，采样negtive shema
                spot_prefix, convert_spot, positive_spot, negative_spot = self.negative_sampler.sample_spot(positive=feature.get('spots', []))
                asoc_prefix, convert_asoc, positive_asoc, negative_asoc = self.negative_sampler.sample_asoc(positive=feature.get('asocs', []))

                # Dynamic generating spot-asoc during training，evaluating时也有
                if 'spot_asoc' in feature:
                    feature['spot_asoc'] = [spot_asoc for spot_asoc in feature['spot_asoc'] if spot_asoc["label"] in positive_spot]
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


            feature.pop('sample_prompt') if 'sample_prompt' in feature else None
            feature.pop('spot_asoc') if 'spot_asoc' in feature else None
            feature['spot'] = [self.tokenizer.encode(s, add_special_tokens = False) for s in spot_prefix]
            feature['asoc'] = [self.tokenizer.encode(a, add_special_tokens = False) for a in asoc_prefix]

            if self.use_ssi:
                prefix = convert_spot + convert_asoc
                if self.max_prefix_length is not None and self.max_prefix_length >= 0:
                    prefix = prefix[:self.max_prefix_length]
                feature['input_ids'] = prefix + [self.negative_sampler.text_start] + feature['input_ids']  # <text>分隔符


            if self.max_length:
                feature['input_ids'] = feature['input_ids'][:self.max_length]
            if self.max_target_length and 'labels' in feature:
                feature['labels'] = feature['labels'][:self.max_target_length]
            
            feature['attention_mask'] = [1] * len(feature['input_ids'])
            if self.max_length:
                feature['input_ids'] = feature['input_ids'] + [self.tokenizer.pad_token_id] * (self.max_length - len(feature['input_ids']))
                feature['attention_mask'] = feature['attention_mask'] + [0] * (self.max_length - len(feature['attention_mask']))


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
        
        return examples        

