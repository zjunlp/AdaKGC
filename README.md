
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
## 🪄 模型

我们的模型tokenizer部分采用了UIE, 其他部分采用t5, 因此是个混合文件, 这里提供了下载链接, 请确保使用这个模型。 [hf_models/mix](https://drive.google.com/file/d/1CI66LlwTWI3qCUCh6InutmrcTxCRrFiK/view?usp=sharing)


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
mode=H
data_name=Few-NERD
task=entity
device=0
ratio=0.8
bash scripts/run_prompt.bash --model=hf_models/mix --data=${data_name}_${mode}/iter_1 --output=${data_name}_${mode}_${ratio} --config=${data_name}.ini --device=${device} --negative_ratio=${ratio} --record2=data/${data_name}_${mode}/iter_7/record.schema

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
mode=H
data_name=NYT
task=relation
device=0
ratio=0.8
bash scripts/run_prompt.bash --model=hf_models/mix --data=${data_name}_${mode}/iter_1 --output=${data_name}_${mode}_${ratio} --config=${data_name}.ini --device=${device} --negative_ratio=${ratio} --record2=data/${data_name}_${mode}/iter_7/record.schema
```

+ ### 事件抽取任务

  <a id="ee"></a>

```bash
mode=H
data_name=ace05_event
task=event
device=0
ratio=0.8
bash scripts/run_prompt.bash --model=hf_models/mix --data=${data_name}_${mode}/iter_1 --output=${data_name}_${mode}_${ratio} --config=${data_name}.ini --device=${device} --negative_ratio=${ratio} --record2=data/${data_name}_${mode}/iter_7/record.schema
```

## 🎰 推理

<a id="inference"></a>

* 仅对单个数据集进行推理（例如`data/ace05_event_H/iter_1`）

```bash
mode=H
data_name=ace05_event
task=event
device=0
ratio=0.8
python3 inference.py --dataname=${data_name}/${data_name}_${mode}/iter_2 --t5_path=hf_models/mix --model=${data_name}_${mode}_${ratio} --task=${task} --cuda=${device} --mode=${mode} --use_prompt --use_ssi --prompt_len=80 --prompt_dim=512
```

`datasetname`: 要预测的数据集的路径(`ace05_event`、`NYT` or `Few-NERD`)。

`model`: 前面训练后得到的模型的路径(训练阶段的output)。

`t5_path`: 基座模型T5(训练阶段的model)。

`task`: 任务类型(entity、relation、event)。

`cuda`: CUDA_VISIBLE_DEVICES。

`mode`: 数据集模式（`H`、`V`、`M`或`R`）。

`use_ssi`、`use_prompt`、`prompt_len`、`prompt_dim`需要跟训练时保持一致, 可以在对应的配置文件config/prompt_conf/ace05_event.ini中查看并设置。


* 在所有迭代数据集上的自动推理（即`data/iter_1/ace05_event_H`~`data/iter _7/ace05_event_H`）

```bash
mode=H
data_name=ace05_event
task=event
device=0
ratio=0.8
python3 inference_mul.py --dataname=${data_name} --t5_path=hf_models/mix --model=${data_name}_${mode}_${ratio} --task=${task} --cuda=${device} --mode=${mode} --use_prompt --use_ssi --prompt_len=80 --prompt_dim=512
```
`use_ssi`、`use_prompt`、`prompt_len`、`prompt_dim`需要跟训练时保持一致。




完整的过程，包括微调和推理（在"scripts/run.bash"中）：

```bash
mode=H
data_name=ace05_event
task=event
device=0
ratio=0.8
bash scripts/run_prompt.bash --model=hf_models/mix --data=${data_name}_${mode}/iter_1 --output=${data_name}_${mode}_${ratio} --config=${data_name}.ini --device=${device} --negative_ratio=${ratio} --record2=data/${data_name}_${mode}/iter_7/record.schema
python3 inference_mul.py --dataname=${task}/${data_name} --t5_path=hf_models/mix --model=${data_name}_${mode}_${ratio} --task=${task} --cuda=${device} --mode=${mode} --use_prompt --use_ssi --prompt_len=80 --prompt_dim=512
python3 inference_mul.py --dataname=${task}/${data_name} --t5_path=hf_models/mix --model=${data_name}_${mode}_${ratio} --task=${task} --cuda=${device} --mode=${mode} --CD --use_prompt --use_ssi --prompt_len=80 --prompt_dim=512
```



| 指标                   | 定义                                                                                      | F1        |
| --------------------- | ---------------------------------------------------------------------------------------- | --------- |
| ent-(P/R/F1)          | 实体的Micro-F1分数(Entity Type, Entity Span)                                                       | spot-F1   |
| rel-strict-(P/R/F1)   | 关系严格模式的Micro-F1分数(Relation Type, Arg1 Span, Arg1 Type, Arg2 Span, Arg2 Type) | asoc-F1 用于关系，spot-F1 用于实体 |
| evt-trigger-(P/R/F1)  | 事件触发词的Micro-F1分数(Event Type, Trigger Span)                                                 | spot-F1   |
| evt-role-(P/R/F1)     | 事件角色的Micro-F1分数 (Event Type, Arg Role, Arg Span)                                            | asoc-F1   |

overall-F1指的是 spot-F1 和 asoc-F1 的总和，可能超100。                                             



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
