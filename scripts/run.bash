mode=R
dataset_name=NYT
task=relation
device=2
ratio=0.8
. config/prompt_conf/${dataset_name}.ini 
bash scripts/run_prompt.bash --model=hf_models/mix --data=data/iter_1/${dataset_name}_${mode} --output=output/baseline-1/${dataset_name} --mode=${mode} --device=${device} --negative_ratio=${ratio} --record2=data/iter_7/${dataset_name}_${mode}/schema.json 
python3 eval/inference_mul.py --dataname=${dataset_name} --model=${output_name} --task=${task} --cuda=${device} --mode=${mode} --use_ssi --prompt_len=50 --prompt_dim=800
python3 eval/inference_mul.py --dataname=${dataset_name} --model=${output_name} --task=${task} --cuda=${device} --mode=${mode} --CD --use_ssi --prompt_len=50 --prompt_dim=800
