# AdaKGC

Code for the paper [`Schema-adaptable Knowledge Graph Construction`](https://arxiv.org/abs/2305.08703).

# Quick Links

* [Requirements](#requirements)
* [Datasets of Extraction Tasks](#datasets-of-extraction-tasks)
* [How to run](#how-to-run)
  * [Named Entity Recognition Task](#ner)
  * [Relation Extraction Task](#re)
  * [Event Extraction Task](#ee)
* [Inference](#inference)
* [Acknowledgment](#acknowledgment)

# Requirements

<a id="requirements"></a>

To run the codes, you need to install the requirements:

```bash
conda create -n adakgc python=3.8
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

```

# Datasets of Extraction Tasks

<a id="datasets-of-extraction-tasks"></a>

Details of dataset construction see [Data Construction](./dataset_construct/README.md).

You can find the dataset as following Google Drive links.

Dataset [ACE05](https://drive.google.com/file/d/14ESd_mjx8PG6E7ls3bxWYuNiPhYWBqlJ/view?usp=sharing)、[Few-NERD](https://drive.google.com/file/d/1K6ZZoJj_FofdqZSLgE6mlHHS3bLWM90Z/view?usp=sharing)、[NYT](https://drive.google.com/file/d/1_x8efbnt5ljaAtUIlqi3T_AVT3nZqoKT/view?usp=sharing)

# How to run

<a id="how-to-run"></a>

```python
mkdir hf_models
cd hf_models
git lfs install
git clone https://huggingface.co/google/t5-v1_1-base
cd ..

mkdir output           # */AdaKGC/output
```

+ ## Named Entity Recognition Task

  <a id="ner"></a>

```bash
# Current path:  */AdaKGC
. config/prompt_conf/Few-NERD_H.ini    # Load predefined parameters
bash scripts/run_finetune.bash --model=hf_models/t5-v1_1-base --data=data/Few-NERD_H/iter_1 --output=output/Few-NERD --mode=H --device=0 --batch=16

```

`model`:  The name or path of a pre-trained model/checkpoint.

`data`: Path to the dataset.

`output`: The path of the fine-tuned checkpoint saved, the final automatically generated output path `AdaKGC/output/ace05_event_H_e30_lr1e-4_b14_n0.8_prompt_80_800`.

`mode`: segmentation mode (`H`、`V`、`M` or `R`).

`device`: CUDA_VISIBLE_DEVICES.

`batch`: batch size.

(see bash scripts and Python files for the detailed command-line arguments)

+ ## Relation Extraction Task

  <a id="re"></a>

```bash
. config/prompt_conf/NYT_H.ini  
bash scripts/run_finetune.bash --model=hf_models/t5-v1_1-base --data=data/NYT_H/iter_1 --output=output/NYT --mode=H --device=0 --batch=16
```

+ ## Event Extraction Task

  <a id="ee"></a>

```bash
. config/prompt_conf/ace05_event_H.ini  
bash scripts/run_finetune.bash --model=hf_models/t5-v1_1-base --data=data/ace05_event_H/iter_1 --output=output/ace05_event --mode=H --device=0 --batch=16
```

# Inference

<a id="inference"></a>

* Only inference on single dataset(`data/ace05_event_H/iter_1`)

```bash
. config/prompt_conf/ace05_event_H.ini
CUDA_VISIBLE_DEVICES=0 python3 eval/inference.py --dataname=data/ace05_event_H/iter_1 --model=output/ace05_event_H_e30_lr1e-4_b16_n0.8_prompt_80_512 --task=event --cuda=0 --mode=H --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}

```

`datasetname`: Path to the datasets which you want to predict.

* Automatic inference on all iteration datasets(`data/iter_1/ace05_event_H` ~ `data/iter_7/ace05_event_H`)

```bash
. config/prompt_conf/ace05_event_H.ini
CUDA_VISIBLE_DEVICES=0 python3 eval/inference_mul.py --dataname=ace05_event --model=output/ace05_event_H_e30_lr1e-4_b16_n0.8_prompt_80_512 --task=event --cuda=0 --mode=H --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}

```

`datasetname`: datasets name(`ace05_event`、`NYT` or `Few-NERD`).

Complete process including fine-tune and inference (in `scripts/run.bash`):

```bash
mode=H
dataset_name=ace05_event
task=event
device=0
output_name=ace05_event_H_e30_lr1e-4_b16_n0.8_prompt_80_512
. config/prompt_conf/${dataset_name}_${mode}.ini 
bash scripts/run_finetune.bash --model=hf_models/mix --data=data/${dataset_name}_${mode}/iter_1 --output=output/${dataset_name} --mode=${mode} --device=${device} 
CUDA_VISIBLE_DEVICES=${device} python3 eval/inference_mul.py --dataname=${dataset_name} --model=${output_name} --task=${task} --mode=${mode} --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}
CUDA_VISIBLE_DEVICES=${device} python3 eval/inference_mul.py --dataname=${dataset_name} --model=${output_name} --task=${task} --mode=${mode} --CD --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}


```

| Metric               | Definition                                                                              |
| -------------------- | --------------------------------------------------------------------------------------- |
| ent-(P/R/F1)         | Micro-F1 of Entity (Entity Type, Entity Span)                                           |
| rel-strict-(P/R/F1)  | Micro-F1 of Relation Strict (Relation Type, Arg1 Span, Arg1 Type, Arg2 Span, Arg2 Type) |
| evt-trigger-(P/R/F1) | Micro-F1 of Event Trigger (Event Type, Trigger Span)                                    |
| evt-role-(P/R/F1)    | Micro-F1 of Event Argument (Event Type, Arg Role, Arg Span)                             |

# Acknowledgment

<a id="acknowledgment"></a>

Part of our code is borrowed from [UIE](https://github.com/universal-ie/UIE) and [UnifiedSKG](https://github.com/hkunlp/unifiedskg), many thanks.

# Papers for the Project & How to Cite

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
