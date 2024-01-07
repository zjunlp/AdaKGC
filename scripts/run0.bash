mode=H
data_name=ace05_event
task=event
device=4
ratio=0.8
bash scripts/run_prompt.bash --model=hf_models/mix --data=data/${data_name}/${data_name}_${mode}/iter_1 --output=output/${data_name}_${mode}_${ratio} --config=${data_name}.ini --device=${device} --negative_ratio=${ratio} --record2=data/${data_name}_${mode}/iter_7/record.schema
python3 inference_mul.py --data=data/${data_name}/${data_name}_${mode} --model=output/${data_name}_${mode}_${ratio} --task=${task} --cuda=${device} --mode=${mode} --use_prompt --use_ssi --prompt_len=80 --prompt_dim=512

