mode=V
dataset_name=ace05_event
task=event
device=0
ratio=0.8
. config/prompt_conf/${dataset_name}.ini 
bash scripts/run_prompt.bash --model=hf_models/mix --data=data/iter_1/${dataset_name}_${mode} --output=output/baseline-1/${dataset_name} --mode=${mode} --device=${device} --negative_ratio=${ratio} --batch=14
