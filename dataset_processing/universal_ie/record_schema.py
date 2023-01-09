#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
from collections import defaultdict
from typing import List


class RecordSchema:
    def __init__(self, type_list, role_list, type_role_dict):
        self.type_list = type_list
        self.role_list = role_list
        self.type_role_dict = type_role_dict

    @staticmethod
    def read_from_file(filename):
        lines = open(filename).readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        type_role_dict = json.loads(lines[2])
        return RecordSchema(type_list, role_list, type_role_dict)

    def write_to_file(self, filename):
        with open(filename, 'w') as output:
            output.write(json.dumps(self.type_list, ensure_ascii=False) + '\n')
            output.write(json.dumps(self.role_list, ensure_ascii=False) + '\n')
            output.write(json.dumps(self.type_role_dict, ensure_ascii=False) + '\n')

