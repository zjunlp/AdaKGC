#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import defaultdict, Counter
from typing import Dict, List
import json
from copy import deepcopy



def tuple_offset(offset):
    if isinstance(offset, tuple):
        return offset
    else:
        return tuple(offset)




class EntityCount:
    def __init__(self):
        self.pred_num = Counter()
        self.gold_num = Counter()
        self.tp = Counter()
        self.notspot = 0
        self.notspan = 0
        self.notproduce = Counter()
        self.mistakeproduce = Counter()


    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b
        

    def compute_spot(self, type_list, new_list):
        result = {}
        result['type_list'] = type_list
        result['new_list'] = new_list
        result['pred_num'] = dict(self.pred_num)
        result['gold_num'] = dict(self.gold_num)
        result['tp'] = dict(self.tp)
        result['notspot'] = self.notspot
        result['notspan'] = self.notspan
        result['notproduce'] = dict(self.notproduce)
        result['mistakeproduce'] = dict(self.mistakeproduce)

        P = {}
        R = {}
        F1 = {}
        notproduce_ratio = {}
        mistakeproduce_ratio = {}

        for t in type_list:
            P[t] = self.safe_div(self.tp.get(t, 0), self.pred_num.get(t, 0)) 
            R[t] = self.safe_div(self.tp.get(t, 0), self.gold_num.get(t, 0))
            F1[t] = self.safe_div(2 * P[t] * R[t], P[t] + R[t]) * 100
            notproduce_ratio[t] = self.safe_div(self.notproduce.get(t, 0), self.gold_num.get(t, 0)) 
            mistakeproduce_ratio[t] = self.safe_div(self.mistakeproduce.get(t, 0), self.pred_num.get(t, 0)) 

        gold_total = sum(self.gold_num.values())
        pred_total = sum(self.pred_num.values())
        tp_total = sum(self.tp.values())

        result['F1_ratio'] = F1
        result['notspot_ratio'] = self.safe_div(self.notspot, pred_total)
        result['notspan_ratio'] = self.safe_div(self.notspan, pred_total)
        result['notproduce_ratio'] = notproduce_ratio
        result['mistakeproduce_ratio'] = mistakeproduce_ratio
        
        P_total_ratio = self.safe_div(tp_total, pred_total)
        R_total_ratio = self.safe_div(tp_total, gold_total)
        result['P_total_ratio'] = P_total_ratio
        result['R_total_ratio'] = R_total_ratio
        result['F1_total_ratio'] = self.safe_div(2 * P_total_ratio * R_total_ratio, P_total_ratio + R_total_ratio) * 100
        result['mistakeproduce_total_ratio'] = self.safe_div(sum(self.mistakeproduce.values()), pred_total)
        result['notproduce_total_ratio'] = self.safe_div(sum(self.notproduce.values()), gold_total)
        
        return result

    

    def count_instance(self, gold_list, pred_list):
        dup_gold_list = deepcopy(gold_list)
        dup_pred_list = deepcopy(pred_list)
        jiaoji = list()

        for gold in gold_list:
            self.gold_num.update([gold[0]])
        for pred in pred_list:
            self.pred_num.update([pred[0]])
            if pred in dup_gold_list:
                self.tp.update([pred[0]])
                jiaoji.append(pred)
                dup_gold_list.remove(pred)
                dup_pred_list.remove(pred)
        
        mistakeproduce = []
        notspot = []
        notspan = []

        for (spot1, span1) in dup_pred_list:
            flag = False
            for (spot2, span2) in dup_gold_list:
                if spot1 == spot2 and span1 != span2:
                    self.notspan += 1
                    notspan.append(((spot1, span1),(spot2, span2)))
                    dup_gold_list.remove((spot2, span2))
                    flag = True
                    break
                elif spot1 != spot2 and span1 == span2:
                    self.notspot += 1
                    notspot.append(((spot1, span1),(spot2, span2)))
                    dup_gold_list.remove((spot2, span2))
                    flag = True
                    break
            if flag == False:
                self.mistakeproduce.update([spot1])
                mistakeproduce.append((spot1, span1))

        notproduce = dup_gold_list
        for (spot, _) in notproduce:
            self.notproduce.update([spot])

        return notspot, notspan, mistakeproduce, notproduce, jiaoji




class RelationCount:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.pred_num = Counter()
        self.gold_num = Counter()
        self.tp = Counter()
        self.nothead = 0
        self.notrelation = 0
        self.nottail = 0
        self.other = 0
        self.notproduce = Counter()
        self.mistakeproduce = Counter()


    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b
        

    def compute_spot(self, type_list, new_list):
        result = {}
        result['type_list'] = type_list
        result['new_list'] = new_list
        result['pred_num'] = self.pred_num
        result['gold_num'] = self.gold_num
        result['tp'] = self.tp
        result[f'not{self.a}'] = self.nothead
        result[f'not{self.b}'] = self.notrelation
        result[f'not{self.c}'] = self.nottail
        result['other'] = self.other
        result['notproduce'] = dict(self.notproduce)
        result['mistakeproduce'] = dict(self.mistakeproduce)

        P = {}
        R = {}
        F1 = {}
        notproduce_ratio = {}
        mistakeproduce_ratio ={}

        for t in type_list:
            P[t] = self.safe_div(self.tp.get(t, 0), self.pred_num.get(t, 0))
            R[t] = self.safe_div(self.tp.get(t, 0), self.gold_num.get(t, 0))
            F1[t] = self.safe_div(2 * P[t] * R[t], P[t] + R[t]) * 100
            notproduce_ratio[t] = self.safe_div(self.notproduce.get(t, 0), self.gold_num.get(t, 0))
            mistakeproduce_ratio[t] = self.safe_div(self.mistakeproduce.get(t, 0), self.pred_num.get(t, 0))


        gold_total = sum(self.gold_num.values())
        pred_total = sum(self.pred_num.values())
        tp_total = sum(self.tp.values())
        result['F1_ratio'] = F1
        result[f'not{self.a}_ratio'] = self.safe_div(self.nothead, pred_total) 
        result[f'not{self.b}_ratio'] = self.safe_div(self.notrelation, pred_total) 
        result[f'not{self.c}_ratio'] = self.safe_div(self.nottail, pred_total)
        result['other_ratio'] = self.safe_div(self.other, pred_total) 
        result['notproduce_ratio'] = notproduce_ratio
        result['mistakeproduce_ratio'] = mistakeproduce_ratio
        
        P_total_ratio = self.safe_div(tp_total, pred_total)
        R_total_ratio = self.safe_div(tp_total, gold_total)
        result['P_total_ratio'] = P_total_ratio
        result['R_total_ratio'] = R_total_ratio
        result['F1_total_ratio'] = self.safe_div(2 * P_total_ratio * R_total_ratio, P_total_ratio + R_total_ratio) * 100
        result['mistakeproduce_total_ratio'] = self.safe_div(sum(self.mistakeproduce.values()), pred_total)
        result['notproduce_total_ratio'] = self.safe_div(sum(self.notproduce.values()), gold_total)

        return result


    def count_instance(self, gold_list, pred_list):
        ''' 关系类型 头实体 尾实体 '''
        dup_gold_list = deepcopy(gold_list)
        dup_pred_list = deepcopy(pred_list)
        jiaoji = list()

        for gold in gold_list:
            self.gold_num.update([gold[0]])
        for pred in pred_list:
            self.pred_num.update([pred[0]])
            if pred in dup_gold_list:
                self.tp.update([pred[0]])
                jiaoji.append(pred)
                dup_gold_list.remove(pred)
                dup_pred_list.remove(pred)

        not_head = []
        not_relation = []
        not_tail = []
        other = []

        for (r, h, t) in dup_pred_list:
            flag = False
            for (r1, h1, t1) in dup_gold_list:
                if r == r1 and h == h1 and t != t1:
                    self.nottail += 1
                    not_tail.append(((r,h,t),(r1,h1,t1)))
                    flag = True
                    break
                elif r == r1 and t == t1 and h != h1:
                    self.nothead += 1
                    not_head.append(((r,h,t),(r1,h1,t1)))
                    flag = True
                    break
                elif h == h1 and t == t1 and r != r1:
                    self.notrelation += 1
                    not_relation.append(((r,h,t),(r1,h1,t1)))
                    flag = True
                    break
            if flag == False:
                self.other += 1
                other.append((r,h,t))
        

        gold_count = {}
        pred_count = {}
        for (relation, _, _) in dup_gold_list:
            if relation not in gold_count.keys():
                gold_count[relation] = 1
            else:
                gold_count[relation] += 1

        for (relation, _, _) in dup_pred_list:
            if relation not in pred_count.keys():
                pred_count[relation] = 1
            else:
                pred_count[relation] += 1
        
        for k, value in gold_count.items():
            c = value - pred_count.get(k, 0)
            self.notproduce.update([k] * max(c, 0))

        for k, value in pred_count.items():
            c = value - gold_count.get(k, 0)
            self.mistakeproduce.update([k] * max(c, 0))


        return not_head, not_relation, not_tail, other, jiaoji





class CaseStudy:
    @staticmethod
    def load_gold_list(gold_list, offset_key=None):
        raise NotImplementedError

    @staticmethod
    def load_pred_list(pred_list):
        raise NotImplementedError

    @staticmethod
    def eval_instance_list(gold_instance_list, pred_instance_list, output_dir, record_schema, new_list):
        raise NotImplementedError




class EntityCase(CaseStudy):
    @staticmethod
    def load_gold_list(gold_list: List[List[Dict]]):
        gold_instance_list = []
        for gold in gold_list:
            gold_offset = list()
            gold_string = list()
            for span in gold:
                span_label = span['type']
                span_offset = span['offset']
                span_text = span['text']
                gold_offset += [(span_label, tuple_offset(span_offset))]
                gold_string += [(span_label, span_text)]
            gold_instance = {
                'offset': gold_offset,
                'string': gold_string,
            }
            gold_instance_list += [gold_instance]
        return gold_instance_list


    @staticmethod
    def load_pred_list(pred_list: List[Dict]):
        pred_instance_list = list()
        for pred in pred_list:
            for offset_pred in pred['offset']:
                if not isinstance(offset_pred[1], tuple):
                    offset_pred[1] = tuple_offset(offset_pred[1])
            pred['offset'] = [tuple_offset(p) for p in pred['offset']]
            pred['string'] = [tuple_offset(p) for p in pred['string']]
            pred_instance_list += [pred]
        return pred_instance_list


    @staticmethod
    def eval_instance_list(gold_instance_list: List[Dict], pred_instance_list: List[Dict], output_dir, record_schema, new_list):
        '''
        最终gold、pred list的格式都是这样的
        [
            {
                'offset': [('Geo-political', (7,)), ('Location', (11,)), ('Geo-political', (14,))],
                'string': [('Geo-political', 'seattle'), ('Location', 'lot'), ('Geo-political', 'city')]
            },
            {...}, ...
        ]
        '''
        log_file = open(f"{output_dir}/log_case.txt", 'w')
        counters = {
            'string': EntityCount(),
        }

        results = []
        for eval_key in counters:
            for pred, gold in zip(pred_instance_list, gold_instance_list):
                notspot, notspan, mistakeproduce, notproduce, tp = counters[eval_key].count_instance(
                    gold_list=gold.get(eval_key, []),
                    pred_list=pred.get(eval_key, [])
                )
                log_file.write(f'gold:\t {json.dumps(gold.get(eval_key, []))}\n')
                log_file.write(f'pred:\t {json.dumps(pred.get(eval_key, []))}\n')
                log_file.write(f'tp:\t {json.dumps(tp)}\n')
                log_file.write(f'notspot:\t {json.dumps(notspot)}\n')
                log_file.write(f'notspan:\t {json.dumps(notspan)}\n')
                log_file.write(f'mistakeproduce:\t {json.dumps(mistakeproduce)}\n')
                log_file.write(f'notproduce :\t {json.dumps(notproduce)}\n')

            result = counters[eval_key].compute_spot(record_schema.type_list, new_list)

            results.append((eval_key, result))
        log_file.close()

        return results




class RelationCase(CaseStudy):
    @staticmethod
    def load_gold_list(gold_list: List[List[Dict]]):
        gold_instance_list = []
        for gold in gold_list:
            gold_instance = defaultdict(list)
            for record in gold:
                assert len(record['args']) == 2
                gold_instance['offset'] += [(
                    record['type'],
                    record['args'][0]['type'],
                    tuple_offset(record['args'][0]['offset']),
                    record['args'][1]['type'],
                    tuple_offset(record['args'][1]['offset']),
                )]
                gold_instance['string'] += [(
                    record['type'],
                    record['args'][0]['type'],
                    record['args'][0]['text'],
                    record['args'][1]['type'],
                    record['args'][1]['text'],
                )]
            gold_instance_list += [gold_instance]

        return gold_instance_list

    @staticmethod
    def load_pred_list(pred_list):
        pred_instance_list = list()
        for pred in pred_list:
            for offset_pred in pred['offset']:

                if not isinstance(offset_pred[2], tuple):
                    offset_pred[2] = tuple_offset(offset_pred[2])

                if not isinstance(offset_pred[4], tuple):
                    offset_pred[4] = tuple_offset(offset_pred[4])

            pred['offset'] = [tuple_offset(p) for p in pred['offset']]
            pred['string'] = [tuple_offset(p) for p in pred['string']]
            pred_instance_list += [pred]
        return pred_instance_list



    @staticmethod
    def eval_instance_list(gold_instance_list: List[Dict], pred_instance_list: List[Dict], output_dir, record_schema, new_list):
        '''
        最终gold、pred list的格式都是这样的
        [
            {
                'offset': [('Part-whole', 'Geo-political', (0,), 'Geo-political', (2,)), ... ],
                'string': [('Part-whole', 'Geo-political', 'MULTAN', 'Geo-political', 'Pakistan'), ...]
                [0]关系类别, [1]头实体类别, [2]头实体, [3]尾实体类别, [4]尾实体 
            }
        ]
        '''
        log_file = open(f"{output_dir}/log_case.txt", 'w')
        # Span Boundary Only    不看标签是否正确，只看句子中对应的Span是否正确
        boundary_counters = {
            'string': RelationCount('head', 'relation', 'tail'),
        }

        results = []
        for eval_key in boundary_counters:
            for pred, gold in zip(pred_instance_list, gold_instance_list):
                nothead, notrelation, nottail, other, tp = boundary_counters[eval_key].count_instance(
                    gold_list=[(x[0], x[2], x[4]) for x in gold.get(eval_key, [])],
                    pred_list=[(x[0], x[2], x[4]) for x in pred.get(eval_key, [])],
                )
                log_file.write(f'gold:\t {json.dumps(gold.get(eval_key, []))}\n')
                log_file.write(f'pred:\t {json.dumps(pred.get(eval_key, []))}\n')
                log_file.write(f'tp:\t {json.dumps(tp)}\n')
                log_file.write(f'nothead:\t {json.dumps(nothead)}\n')
                log_file.write(f'notrelation:\t {json.dumps(notrelation)}\n')
                log_file.write(f'nottail:\t {json.dumps(nottail)}\n')
                log_file.write(f'other:\t {json.dumps(other)}\n')

            result = boundary_counters[eval_key].compute_spot(record_schema.role_list, new_list)
            results.append((eval_key, result))
        log_file.close()

        return results



class EventCase(CaseStudy):
    @staticmethod
    def load_gold_list(gold_list):
        gold_instance_list = []
        for gold in gold_list:
            gold_instance = defaultdict(list)
            for record in gold:
                gold_instance['offset_trigger'] += [(record['type'], tuple_offset(record['offset']))]
                gold_instance['string_trigger'] += [(record['type'], record['text'])]
                for arg in record['args']:
                    gold_instance['offset_role'] += [(record['type'], arg['type'], tuple_offset(arg['offset']))]
                    gold_instance['string_role'] += [(record['type'], arg['type'], arg['text'])]
            gold_instance_list += [gold_instance]
        return gold_instance_list

    @staticmethod
    def load_pred_list(pred_list):
        pred_instance_list = list()
        for pred in pred_list:
            pred_instance = defaultdict(list)

            for offset_pred in pred['offset']:
                event_type, trigger_offset = offset_pred['type'], tuple_offset(offset_pred['trigger'])
                pred_instance['offset_trigger'] += [(event_type, trigger_offset)]
                for role_type, role_offset in offset_pred['roles']:
                    pred_instance['offset_role'] += [(event_type, role_type, tuple_offset(role_offset))]

            for string_pred in pred['string']:
                event_type, trigger_string = string_pred['type'], string_pred['trigger']
                pred_instance['string_trigger'] += [(event_type, trigger_string)]
                for role_type, role_string in string_pred['roles']:
                    pred_instance['string_role'] += [(event_type, role_type, role_string)]
            pred_instance_list += [pred_instance]
        return pred_instance_list

    @staticmethod
    def eval_instance_list(gold_instance_list: List[Dict], pred_instance_list: List[Dict], output_dir, record_schema, new_list):
        """
        gold_instance_list (List[Dict]): List of Sentece, each sentence contains Four List of Event Tuple
            [
                {
                    'offset_trigger': [('Die', (16,)), ('Convict', (30,))],
                    'string_trigger': [('Die', 'shot'), ('Convict', 'convicted')],
                    'offset_role': [('Die', 'Victim', (17,)), ('Die', 'Agent', (5, 6)), ('Die', 'Place', (23,))],
                    'string_role': [('Die', 'Victim', 'himself'), ('Die', 'Agent', 'John Joseph'), ('Die', 'Place', 'court')]
                },
                ...
            ]
        """
        log_file = open(f"{output_dir}/log_case.txt", 'w')
        # Span Boundary Only    不看标签是否正确，只看句子中对应的Span是否正确
        trigger_counters = {
            'string_trigger': EntityCount(),
            'string_role': RelationCount('spot','asoc','span'),
        }

        results = []
        for eval_key in trigger_counters:
            for pred, gold in zip(pred_instance_list, gold_instance_list):
                if eval_key == 'string_role':
                    notspot, notasoc, notspan, other, tp = trigger_counters[eval_key].count_instance(
                        gold_list=[(x[0], x[1], x[2]) for x in gold.get(eval_key, [])],
                        pred_list=[(x[0], x[1], x[2]) for x in pred.get(eval_key, [])],
                    )
                    log_file.write(f'gold:\t {json.dumps(gold.get(eval_key, []))}\n')
                    log_file.write(f'pred:\t {json.dumps(pred.get(eval_key, []))}\n')
                    log_file.write(f'tp:\t {json.dumps(tp)}\n')
                    log_file.write(f'notspot:\t {json.dumps(notspot)}\n')
                    log_file.write(f'notasoc:\t {json.dumps(notasoc)}\n')
                    log_file.write(f'notspan:\t {json.dumps(notspan)}\n')
                    log_file.write(f'other:\t {json.dumps(other)}\n')
                    result = trigger_counters[eval_key].compute_spot(record_schema.role_list, new_list)
                else:
                    notspot, notspan, mistakeproduce, notproduce, tp = trigger_counters[eval_key].count_instance(
                        gold_list=[(x[0], x[1]) for x in gold.get(eval_key, [])],
                        pred_list=[(x[0], x[1]) for x in pred.get(eval_key, [])],
                    )
                    log_file.write(f'gold:\t {json.dumps(gold.get(eval_key, []))}\n')
                    log_file.write(f'pred:\t {json.dumps(pred.get(eval_key, []))}\n')
                    log_file.write(f'tp:\t {json.dumps(tp)}\n')
                    log_file.write(f'notspot:\t {json.dumps(notspot)}\n')
                    log_file.write(f'notspan:\t {json.dumps(notspan)}\n')
                    log_file.write(f'mistakeproduce:\t {json.dumps(mistakeproduce)}\n')
                    log_file.write(f'notproduce :\t {json.dumps(notproduce)}\n')
                    result = trigger_counters[eval_key].compute_spot(record_schema.type_list, new_list)

            results.append((eval_key, result))
        log_file.close()

        return results

