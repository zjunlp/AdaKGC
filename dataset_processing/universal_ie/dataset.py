#!/usr/bin/env python
# -*- coding:utf-8 -*-
from universal_ie.utils import label_format
import yaml
import os
from typing import Dict, List
import universal_ie.task_format as task_format


class Dataset:
    def __init__(self, name: str, path: str, data_class: task_format.TaskFormat, split_dict: Dict, language: str, mapper: Dict, delete_list: List, other: Dict = None) -> None:
        self.name = name
        self.path = path
        self.data_class = data_class
        self.split_dict = split_dict
        self.language = language
        self.mapper = mapper
        self.other = other
        self.delete_list = delete_list

    def load_dataset(self, logger_name):
        datasets = {}
        for split_name, filename in self.split_dict.items():
            datasets[split_name] = self.data_class.load_from_file(
                filename=os.path.join(self.path, filename),
                language=self.language,
                delete_list=self.delete_list,
                m=self.mapper,
                logger_name=logger_name,
                **self.other,
            )
        return datasets

    @staticmethod
    def load_yaml_file(yaml_file):
        dataset_config = yaml.load(open(yaml_file), Loader=yaml.FullLoader)
        if 'mapper' in dataset_config:
            mapper = dataset_config['mapper']
            for key in mapper:
                mapper[key] = label_format(mapper[key])
        else:
            print(f"{dataset_config['name']} without label mapper.")
            mapper = None

        return Dataset(
            name=dataset_config['name'],  
            path=dataset_config['path'],  
            data_class=getattr(task_format, dataset_config['data_class']),  
            split_dict=dataset_config['split'],   
            language=dataset_config['language'],  
            mapper=mapper,   
            delete_list=dataset_config['delete_list'] if dataset_config['delete_list'] is not None else [],
            other=dataset_config.get('other', {}),
        )
