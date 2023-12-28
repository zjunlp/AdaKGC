
<h1 align="center"> 🎇AdaKGC 
</h1>
<div align="center">
     
   [![Awesome](https://awesome.re/badge.svg)]() 
   [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
   ![](https://img.shields.io/github/last-commit/zjunlp/AdaKGC?color=green) 
   ![](https://img.shields.io/badge/PRs-Welcome-red) 
</div>

## *👋 新闻!*

- 论文代码[`Schema-adaptable Knowledge Graph Construction`](https://arxiv.org/abs/2305.08703).

- 我们的工作已被EMNLP2023 Findings会议接受。


## 🎉 快速链接

- [*👋 新闻!*](#-新闻)
- [🎉 快速链接](#-快速链接)
- [🎈 环境依赖](#-环境依赖)
- [🎏 数据集](#-数据集)
- [⚾ 运行](#-运行)
- [🎰 推理](#-推理)
- [🏳‍🌈 Acknowledgment](#-acknowledgment)
- [🚩 Papers for the Project \& How to Cite](#-papers-for-the-project--how-to-cite)

## 🎈 环境依赖

<a id="requirements"></a>

要运行代码，您需要安装以下要求:

```bash
conda create -n adakgc python=3.8
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

```



## 🎏 数据集

<a id="datasets-of-extraction-tasks"></a>

数据集构造的详细信息请参见[Data Construction](./dataset_construct/README.md).

您可以通过以下Google Drive链接找到数据集。

Dataset [ACE05](https://drive.google.com/file/d/14ESd_mjx8PG6E7ls3bxWYuNiPhYWBqlJ/view?usp=sharing)、[Few-NERD](https://drive.google.com/file/d/1K6ZZoJj_FofdqZSLgE6mlHHS3bLWM90Z/view?usp=sharing)、[NYT](https://drive.google.com/file/d/1_x8efbnt5ljaAtUIlqi3T_AVT3nZqoKT/view?usp=sharing)

## ⚾ 运行

<a id="how-to-run"></a>

```python
mkdir hf_models
cd hf_models
git lfs install
git clone https://huggingface.co/google/t5-v1_1-base
cd ..

mkdir output           # */AdaKGC/output
```

+ ### 实体识别任务

  <a id="ner"></a>

```bash
# Current path:  */AdaKGC
. config/prompt_conf/Few-NERD_H.ini    # Load predefined parameters
bash scripts/run_finetune.bash --model=hf_models/t5-v1_1-base --data=data/Few-NERD_H/iter_1 --output=output/Few-NERD --mode=H --device=0 --batch=16
```

`model`: 预训练的模型的名称或路径。
`data`: 数据集的路径。
`output`: 保存的微调检查点的路径，最终自动生成的输出路径`AdaKGC/output/ace05_event_H_e30_lr1e-4_b14_n0。
`mode`: 数据集模式（`H`、`V`、`M`或`R`）。
`device`: CUDA_VISIBLE_DEVICES。
`batch`: batch size。
（有关详细的命令行参数，请参阅bash脚本和Python文件）




+ ### 关系抽取任务

  <a id="re"></a>

```bash
. config/prompt_conf/NYT_H.ini  
bash scripts/run_finetune.bash --model=hf_models/t5-v1_1-base --data=data/NYT_H/iter_1 --output=output/NYT --mode=H --device=0 --batch=16
```

+ ### 事件抽取任务

  <a id="ee"></a>

```bash
. config/prompt_conf/ace05_event_H.ini  
bash scripts/run_finetune.bash --model=hf_models/t5-v1_1-base --data=data/ace05_event_H/iter_1 --output=output/ace05_event --mode=H --device=0 --batch=16
```

## 🎰 推理

<a id="inference"></a>

* 仅对单个数据集进行推理（例如`data/ace05_event_H/iter_1`）

```bash
. config/prompt_conf/ace05_event_H.ini
CUDA_VISIBLE_DEVICES=0 python3 eval/inference.py --dataname=data/ace05_event_H/iter_1 --model=output/ace05_event_H_e30_lr1e-4_b16_n0.8_prompt_80_512 --t5_path=hf_models/t5-v1_1-base --task=event --cuda=0 --mode=H --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}

```

`datasetname`: 要预测的数据集的路径(`ace05_event`、`NYT` or `Few-NERD`)。
`model`: 前面训练后得到的模型的路径(训练阶段的output)。
`t5_path`: 基座模型T5(训练阶段的model)。
`task`: 任务类型(entity、relation、event)。
`cuda`: CUDA_VISIBLE_DEVICES。
`mode`: 数据集模式（`H`、`V`、`M`或`R`）。
`use_ssi`、`use_task`、`use_prompt`、`prompt_len`、`prompt_dim`需要跟训练时保持一致。


* 在所有迭代数据集上的自动推理（即`data/iter_1/ace05_event_H`~`data/iter _7/ace05_event_H`）

```bash
. config/prompt_conf/ace05_event_H.ini
CUDA_VISIBLE_DEVICES=0 python3 eval/inference_mul.py --dataname=ace05_event --model=output/ace05_event_H_e30_lr1e-4_b16_n0.8_prompt_80_512 --task=event --cuda=0 --mode=H --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}

```


完整的过程，包括微调和推理（在"scripts/run.bash"中）：

```bash
mode=H
dataset_name=ace05_event
task=event
device=0
output_name=ace05_event_H_e30_lr1e-4_b16_n0.8_prompt_80_512
. config/prompt_conf/${dataset_name}_${mode}.ini 
bash scripts/run_finetune.bash --model=hf_models/t5-v1_1-base --data=data/${dataset_name}_${mode}/iter_1 --output=output/${dataset_name} --mode=${mode} --device=${device} 
CUDA_VISIBLE_DEVICES=${device} python3 eval/inference_mul.py --dataname=${dataset_name} --t5_model=hf_models/t5-v1_1-base --model=${output_name} --task=${task} --mode=${mode} --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}
CUDA_VISIBLE_DEVICES=${device} python3 eval/inference_mul.py --dataname=${dataset_name} --t5_model=hf_models/t5-v1_1-base --model=${output_name} --task=${task} --mode=${mode} --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}
```

| Metric               | Definition                                                                              | F1      |
| -------------------- | --------------------------------------------------------------------------------------- |-------|
| ent-(P/R/F1)         | Micro-F1 of Entity (Entity Type, Entity Span)                                           | spot-F1      |
| rel-strict-(P/R/F1)  | Micro-F1 of Relation Strict (Relation Type, Arg1 Span, Arg1 Type, Arg2 Span, Arg2 Type) | asoc-F1 for relation、spot-F1 for entity     |
| evt-trigger-(P/R/F1) | Micro-F1 of Event Trigger (Event Type, Trigger Span)                                    | spot-F1      |
| evt-role-(P/R/F1)    | Micro-F1 of Event Argument (Event Type, Arg Role, Arg Span)                             | asoc-F1      |

overall-F1 refer to the sum of spot-F1 and asoc-F1, which may over 100.

## 🏳‍🌈 Acknowledgment

<a id="acknowledgment"></a>

Part of our code is borrowed from [UIE](https://github.com/universal-ie/UIE) and [UnifiedSKG](https://github.com/hkunlp/unifiedskg), many thanks.

## 🚩 Papers for the Project & How to Cite

If you use or extend our work, please cite the paper as follows:

```bibtex
@article{DBLP:journals/corr/abs-2305-08703,
  author       = {Hongbin Ye and
                  Honghao Gui and
                  Xin Xu and
                  Huajun Chen and
                  Ningyu Zhang},
  title        = {Schema-adaptable Knowledge Graph Construction},
  journal      = {CoRR},
  volume       = {abs/2305.08703},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.08703},
  doi          = {10.48550/arXiv.2305.08703},
  eprinttype    = {arXiv},
  eprint       = {2305.08703},
  timestamp    = {Wed, 17 May 2023 15:47:36 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-08703.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
