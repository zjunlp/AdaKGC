#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import defaultdict
from adakgc.utils.constants import BaseStructureMarker


def convert_spot_asoc(spot_asoc_instance, structure_maker):
    spot_instance_str_rep_list = list()
    for spot in spot_asoc_instance:
        spot_str_rep = [
            spot['label'],
            structure_maker.target_span_start,
            spot['span'],
        ]     
        for asoc_label, asoc_span in spot.get('asoc', list()):
            asoc_str_rep = [
                structure_maker.span_start,
                asoc_label,
                structure_maker.target_span_start,
                asoc_span,
                structure_maker.span_end,
            ]      
            spot_str_rep += [' '.join(asoc_str_rep)]
        spot_instance_str_rep_list += [' '.join([
            structure_maker.record_start,
            ' '.join(spot_str_rep),
            structure_maker.record_end,
        ])]  
    target_text = ' '.join([
        structure_maker.sent_start,
        ' '.join(spot_instance_str_rep_list),
        structure_maker.sent_end,
    ])   
    return target_text





def text2spotasoc(entities, relations, events):
    """Convert Entity Relation Event to Spot-Asoc
    """
    spot_dict = dict()
    asoc_dict = defaultdict(list)

    def add_spot(spot):
        spot_key = (tuple(spot["offset"]), spot["type"]) 
        spot_dict[spot_key] = spot  

    def add_asoc(spot, asoc, tail):
        spot_key = (tuple(spot["offset"]), spot["type"])
        asoc_dict[spot_key] += [(tail["offset"], tail, asoc)]   
        

    for entity in entities:
        add_spot(spot=entity)

    for relation in relations:
        add_spot(spot=relation["args"][0])
        add_asoc(spot=relation["args"][0], asoc=relation["type"], tail=relation["args"][1])

    for event in events:
        add_spot(spot=event)
        for arg in event["args"]:
            add_asoc(spot=event, asoc=arg["type"], tail=arg)

    spot_asoc_instance = list()
    for spot_key in sorted(spot_dict.keys()):
        _, label = spot_key

        if spot_dict[spot_key]["text"] == "":
            continue

        spot_instance = {'span': spot_dict[spot_key]["text"],
                            'label': label,
                            'asoc': list(),
                        }

        for _, tail, asoc in asoc_dict.get(spot_key, []):
            if tail["text"] == "":
                continue
            spot_instance['asoc'] += [(asoc, tail["text"])]
        spot_asoc_instance += [spot_instance]

    target_text = convert_spot_asoc(
        spot_asoc_instance,
        structure_maker=BaseStructureMarker(),
    )

    spot_labels = set([label for _, label in spot_dict.keys()])
    asoc_labels = set()
    for _, asoc_list in asoc_dict.items():
        for _, _, asoc in asoc_list:
            asoc_labels.add(asoc)
            
    return target_text, list(spot_labels), list(asoc_labels), spot_asoc_instance

