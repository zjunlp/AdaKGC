CUDA_VISIBLE_DEVICES=0
model_name="hf_models/mix"
data_name="data/relation/NYT_H"
output_name="output/NYT_H_prompt"
config_name="config/prompt_conf/NYT.ini"

OPTS=$(getopt -o m:d:o:c --long batch:,device:,output:,mode:,model:,data:,task_name:,seed:,lr_rate:,lr_scheduler:,label_smoothing:,epoch:,format:,eval_steps:,warmup_ratio:,constraint_decoding,verbose,preprocess,fp16:,negative:,random_prompt,max_source_length:,max_target_length:,spot_noise:,asoc_noise:,positive:,map_config:,negative_ratio:,use_ema:,prompt_dim:,prompt_len:,use_ssi:,use_prompt:,use_task:,init_prompt:,freeze_LM:,record2:, -n 'parse-options' -- "$@")
if [ $? != 0 ]; then
  echo "Failed parsing options." >&2
  exit 1
fi
eval set -- "$OPTS"


while true; 
do
  case "$1" in
  --batch)
    batch_size="$2"
    shift
    shift
    ;;
  --device)
    CUDA_VISIBLE_DEVICES="$2"
    shift
    shift
    ;;
  --output)
    output="$2"
    shift
    shift
    ;;
  --mode)
    mode="$2"
    shift
    shift
    ;;
  --model)
    model_name="$2"
    shift
    shift
    ;;
  --data)
    data_name="$2"
    shift
    shift
    ;;
  --task_name)
    task_name="$2"
    shift
    shift
    ;;
  --seed)
    seed="$2"
    shift
    shift
    ;;
  --lr_rate)
    lr_rate="$2"
    shift
    shift
    ;;
  --format)
    decoding_format="$2"
    shift
    shift
    ;;
  --negative)
    negative="$2"
    shift
    shift
    ;;
  --positive)
    positive="$2"
    shift
    shift
    ;;
  --lr_scheduler)
    lr_scheduler="$2"
    shift
    shift
    ;;
  --label_smoothing)
    label_smoothing="$2"
    shift
    shift
    ;;
  --epoch)
    epoch="$2"
    shift
    shift
    ;;
  --eval_steps)
    eval_steps="$2"
    shift
    shift
    ;;
  --warmup_ratio)
    warmup_ratio="$2"
    shift
    shift
    ;;
  --max_source_length)
    max_source_length="$2"
    shift
    shift
    ;;
  --max_target_length)
    max_target_length="$2"
    shift
    shift
    ;;
  --spot_noise)
    spot_noise="$2"
    shift
    shift
    ;;
  --asoc_noise)
    asoc_noise="$2"
    shift
    shift
    ;;
  --fp16)
    fp16="$2"
    shift
    shift
    ;;
  --map_config)
    map_config="$2"
    shift
    shift
    ;;
  --constraint_decoding)
    constraint_decoding="--constraint_decoding"
    shift
    ;;
  --preprocess)
    preprocess=True
    shift
    ;;
  --random_prompt)
    ordered_prompt=False
    shift
    ;;
  --verbose)
    verbose=true
    shift
    ;;
  --negative_ratio)
    negative_ratio="$2"
    shift
    shift
    ;;
  --use_ema)
    use_ema="$2"
    shift
    shift
    ;;
  --prompt_dim)
    prompt_dim="$2"
    shift
    shift
    ;;
  --prompt_len)
    prompt_len="$2"
    shift
    shift
    ;;
  --use_ssi)
    use_ssi="$2"
    shift
    shift
    ;;
  --use_prompt)
    use_prompt="$2"
    shift
    shift
    ;;
  --use_task)
    use_task="$2"
    shift
    shift
    ;;
  --init_prompt)
    init_prompt="$2"
    shift
    shift
    ;;
  --freeze_LM)
    freeze_LM="$2"
    shift
    shift
    ;;
  --record2)
    record2="$2"
    shift
    shift
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "$1" not recognize.
    exit
    ;;
  esac
done


export PYTHONPATH="${PYTHONPATH}:./"
output_name=${output}_${mode}_e${epoch}_lr${lr_rate}_b${batch_size}_n${negative_ratio}
if [[ ${use_prompt} == True ]]
then
  output_name=${output_name}_prompt_${prompt_len}_${prompt_dim}
fi
if [[ ${use_task} == True ]]
then
  output_name=${output_name}_task
fi
echo ${output_name}
export output_name


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 run.py \
    --do_train --do_eval --do_predict ${constraint_decoding} ${fp16} \
    --use_fast_tokenizer=True \
    --from_checkpoint=True \
    --ddp_find_unused_parameters=False \
    --predict_with_generate \
    --evaluation_strategy=${evaluation_strategy:-"epoch"} \
    --save_strategy=${save_strategy:-"epoch"} \
    --metric_for_best_model "eval_overall-F1" \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --max_source_length=${max_source_length:-"256"} \
    --max_prefix_length=${max_prefix_length:-"-1"} \
    --max_target_length=${max_target_length:-"192"} \
    --num_train_epochs=${epoch} \
    --train_file=${data_name}/train.json \
    --validation_file=${data_name}/val.json \
    --test_file=${data_name}/test.json \
    --record_schema=${data_name}/schema.json \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=$((batch_size * 4)) \
    --output_dir=${output_name} \
    --overwrite_output_dir=False\
    --logging_dir=${output_name}_log \
    --model_name_or_path=${model_name} \
    --learning_rate=${lr_rate} \
    --lr_scheduler_type=${lr_scheduler} \
    --label_smoothing_factor=${label_smoothing} \
    --decoding_format ${decoding_format} \
    --warmup_ratio ${warmup_ratio} \
    --preprocessing_num_workers=4 \
    --dataloader_num_workers=0 \
    --meta_negative=${negative} \
    --meta_positive_rate=${positive} \
    --skip_memory_metrics \
    --no_remove_unused_columns \
    --ordered_prompt=${ordered_prompt} \
    --save_better_checkpoint=False \
    --start_eval_step=${start_eval_step:-"0"} \
    --spot_noise=${spot_noise} \
    --asoc_noise=${asoc_noise} \
    --negative_ratio=${negative_ratio:-"1"} \
    --task_name=${task_name} \
    --use_ema=${use_ema:-"False"} \
    --prompt_len=${prompt_len} \
    --prompt_dim=${prompt_dim} \
    --use_ssi=${use_ssi:-"True"} \
    --use_task=${use_task:-"True"} \
    --init_prompt=${init_prompt:-"True"} \
    --freeze_LM=${freeze_LM:-"False"} \
    --record2=${record2:-""} \
    --seed=${seed} --disable_tqdm=${disable_tqdm}
