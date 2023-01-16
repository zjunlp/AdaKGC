mode=H
dataset_name=ace05_event
task=event
device=1
. config/prompt_conf/${dataset_name}_${mode}.ini 
bash scripts/run_finetune.bash --model=hf_models/mix --data=data/iter_1/${dataset_name}_${mode} --output=output/${dataset_name} --mode=${mode} --device=${device} 
CUDA_VISIBLE_DEVICES=${device} python3 eval/inference_mul.py --dataname=${dataset_name} --model=${output_name} --task=${task} --mode=${mode} --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}
CUDA_VISIBLE_DEVICES=${device} python3 eval/inference_mul.py --dataname=${dataset_name} --model=${output_name} --task=${task} --mode=${mode} --CD --use_ssi=${use_ssi} --use_task=${use_task} --use_prompt=${use_prompt} --prompt_len=${prompt_len} --prompt_dim=${prompt_dim}
