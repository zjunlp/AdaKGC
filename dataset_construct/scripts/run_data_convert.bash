#!/usr/bin/env bash
# -*- coding:utf-8 -*-

#python data_convert.py --config "../config/data_config/event" --iter_num=7 --task="ace05_event" --mode="H"
#python data_convert.py --config "../config/data_config/event" --iter_num=7 --task="ace05_event" --mode="V"
#python data_convert.py --config "../config/data_config/event" --iter_num=7 --task="ace05_event" --mode="M"
#python data_convert.py --config "../config/data_config/event" --iter_num=7 --task="ace05_event" --mode="R"

python data_convert.py --config "../config/data_config/relation" --iter_num=7 --task="NYT" --mode="H"
python data_convert.py --config "../config/data_config/relation" --iter_num=7 --task="NYT" --mode="V"
python data_convert.py --config "../config/data_config/relation" --iter_num=7 --task="NYT" --mode="M"
python data_convert.py --config "../config/data_config/relation" --iter_num=7 --task="NYT" --mode="R"

python data_convert.py --config "../config/data_config/entity" --iter_num=7 --task="Few-NERD" --mode="H"
python data_convert.py --config "../config/data_config/entity" --iter_num=7 --task="Few-NERD" --mode="V"
python data_convert.py --config "../config/data_config/entity" --iter_num=7 --task="Few-NERD" --mode="M"
python data_convert.py --config "../config/data_config/entity" --iter_num=7 --task="Few-NERD" --mode="R"
