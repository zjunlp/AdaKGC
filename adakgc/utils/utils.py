#!/usr/bin/env python
# -*- coding:utf-8 -*-
import re
def convert_spot_asoc(spot_asoc_instance, structure_maker):
    '''
    Convert spot-asoc to SEL
    '''
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


def convert_spot_asoc_name(spot_asoc_instance, structure_maker):
    """
    Args:
        spot_asoc_instance ([type]): [description]
        structure_maker ([type]): [description]

    Returns:
        [type]: [description]
    """
    spot_instance_str_rep_list = list()
    for spot in spot_asoc_instance:
        spot_str_rep = [
            spot['span'],
            structure_maker.target_span_start,
            spot['label'],
        ]
        for asoc_label, asoc_span in spot.get('asoc', list()):
            asoc_str_rep = [
                structure_maker.span_start,
                asoc_span,
                structure_maker.target_span_start,
                asoc_label,
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


convert_to_record_function = {
    'spotasoc': convert_spot_asoc,
    'spotasocname': convert_spot_asoc_name,
}


def fix_unk_from_text(span, text, unk='<unk>'):
    """
    Find span from the text to fix unk in the generated span

    Example:
    span = "<unk> colo e Bengo"
    text = "At 159 meters above sea level , Angola International Airport is located at Ícolo e Bengo , part of Luanda Province , in Angola ."
    """
    if unk not in span:
        return span

    def clean_wildcard(x):
        sp = ".*?()[]+"
        return re.sub("("+"|".join([f"\\{s}" for s in sp])+")", "\\\\\g<1>", x)

    match = r'\s*\S+\s*'.join([clean_wildcard(item.strip()) for item in span.split(unk)])

    result = re.search(match, text)

    if not result:
        return span
    return result.group().strip()