- Code for [`Schema-adaptable Knowledge Graph Construction`]()

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

`<a id="requirements"></a>`

To run the codes, you need to install the requirements:

```bash
conda create -n adakgc python=3.8
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

```

# Datasets of Extraction Tasks

`<a id="datasets-of-extraction-tasks"></a>`

Details of dataset construction see [Data Construction](./dataset_construct/README.md).

You can find the dataset as following Google Drive links.

Dataset [[Google Drive]]()

# How to run

`<a id="how-to-run"></a>`

```python
mkdir hf_models
cd hf_models
git lfs install
git clone https://huggingface.co/google/t5-v1_1-base
cd ..

mkdir output           # */AdaKGC/output
```

+ ## Named Entity Recognition Task

  `<a id="ner"></a>`

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

  `<a id="re"></a>`

```bash
. config/prompt_conf/NYT_H.ini  
bash scripts/run_finetune.bash --model=hf_models/t5-v1_1-base --data=data/NYT_H/iter_1 --output=output/NYT --mode=H --device=0 --batch=16
```

+ ## Event Extraction Task

  `<a id="ee"></a>`

```bash
. config/prompt_conf/ace05_event_H.ini  
bash scripts/run_finetune.bash --model=hf_models/t5-v1_1-base --data=data/ace05_event_H/iter_1 --output=output/ace05_event --mode=H --device=0 --batch=16
```

# Inference

`<a id="inference"></a>`

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

Complete process (in `scripts/run.bash`):

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

`<a id="acknowledgment"></a>`

Part of our code is borrowed from [UIE](https://github.com/universal-ie/UIE) and [UnifiedSKG](https://github.com/hkunlp/unifiedskg), many thanks.

# Papers for the Project & How to Cite

If you use or extend our work, please cite the paper as follows:

```bibtex

```
