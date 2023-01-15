- Code for [`Schema-adaptable Knowledge Graph Construction`]()


# Requirements

To run the codes, you need to install the requirements:

``` bash
conda create -n adakgc python=3.8
pip install -r requirements.txt

```



# Datasets of Extraction Tasks

Details of dataset construction see [Data Construction](dataset_construct/README.md).

You can find the dataset as following Google Drive links.

Dataset [[Google Drive]]()



# How to run

```python
mkdir output           # */AdaKGC/output
```

+ ## Named Entity Recognition Task

```bash
# Current path:  */AdaKGC
. config/prompt_conf/Few-NERD.ini    # Load predefined parameters
bash scripts/run_prompt.bash --model=hf_models/t5-v1_1-base --data=data/iter_1/Few-NERD_H --output=output/Few-NERD --mode=H --device=0 --batch=16

```

`model`:  The name or path of a pre-trained model/checkpoint.

`data`: Path to dataset.

`output`: The path of the fine-tuned checkpoint saved, the final automatically generated output path `AdaKGC/output/ace05_event_H_e30_lr1e-4_b14_n0.8_prompt_80_800`.

`mode`: segmentation mode (`H`、`V`、`M` or `R`).

`device`: CUDA_VISIBLE_DEVICES.

`batch`: batch size.

(see bash scripts and Python files for the detailed command-line arguments)

+ ## Relation Extraction Task


```bash
. config/prompt_conf/NYT.ini    
bash scripts/run_prompt.bash --model=hf_models/t5-v1_1-base --data=data/iter_1/NYT_H --output=output/NYT --mode=H --device=0 --batch=16
```



+ ## Event Extraction Task

```bash
. config/prompt_conf/ace05_event.ini    
bash scripts/run_prompt.bash --model=hf_models/t5-v1_1-base --data=data/iter_1/ace05_event_H --output=output/ace05_event --mode=H --device=0 --batch=16
```



# Inference

* Only inference on single dataset(`data/iter_1/ace05_event_H`)

```bash
. config/prompt_conf/ace05_event.ini
python3 eval/inference.py --dataname=data/iter_1/ace05_event_H --model=output/ace05_event_H_e30_lr1e-4_b14_n0.8_prompt_80_800 --task=event --cuda=0 --mode=H --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}

```

`datasetname`: Path to the datasets which you want to inference.

* Automatic inference on all iteration datasets(`data/iter_1/ace05_event_H` ~ `data/iter_7/ace05_event_H`)

```bash
. config/prompt_conf/ace05_event.ini
python3 eval/inference.py --dataname=ace05_event --model=output/ace05_event_H_e30_lr1e-4_b14_n0.8_prompt_80_800 --task=event --cuda=0 --mode=H --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}

```

`datasetname`: datasets name(`ace05_event`、`NYT` or `Few-NERD`).

| Metric      | Definition |
| ----------- | ----------- |
| ent-(P/R/F1)      | Micro-F1 of Entity (Entity Type, Entity Span) |
| rel-strict-(P/R/F1)   | Micro-F1 of Relation Strict (Relation Type, Arg1 Span, Arg1 Type, Arg2 Span, Arg2 Type) |
| rel-boundary-(P/R/F1)   | Micro-F1 of Relation Boundary (Relation Type, Arg1 Span, Arg2 Span) |
| evt-trigger-(P/R/F1)   | Micro-F1 of Event Trigger (Event Type, Trigger Span) |
| evt-role-(P/R/F1)   | Micro-F1 of Event Argument (Event Type, Arg Role, Arg Span) |



# Acknowledgement

Part of our code is borrowed from [code](https://github.com/universal-ie/UIE) of [Unified Structure Generation for Universal Information Extraction](https://arxiv.org/abs/2203.12277), many thanks.

# Papers for the Project & How to Cite

If you use or extend our work, please cite the paper as follows:

```bibtex

```

