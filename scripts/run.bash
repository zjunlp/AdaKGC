mode=R
data_name=Few-NERD
task=entity
device=3
ratio=0.8
#bash scripts/run_prompt.bash --model=hf_models/mix --data=iter_1/${data_name}_${mode} --output=output/baseline-1/${data_name}_${mode}_${ratio} --config=${data_name}.ini --device=${device} --negative_ratio=${ratio} --record2=/zjunlp/ghh/UIE/data/iter_7/${data_name}_${mode}/record.schema
python3 eval/inference_prompt_mul.py --dataname=${data_name} --model=output/baseline-1/${data_name}_${mode}_${ratio}_prompt_prefix --task=${task} --cuda=${device} --mode=${mode} --use_ssi --prompt_len=80 --prompt_dim=800
python3 eval/inference_prompt_mul.py --dataname=${data_name} --model=output/baseline-1/${data_name}_${mode}_${ratio}_prompt_prefix --task=${task} --cuda=${device} --mode=${mode} --CD --use_ssi --prompt_len=80 --prompt_dim=800
