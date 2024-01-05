# 默认参数
export CUDA_VISIBLE_DEVICES="0"
export task="meta"
export decoding_format='spotasoc'
export start_eval_step=500
export constraint_decoding=''
export verbose=False
export fp16=''
export positive=1
export ordered_prompt=True
export seed="42"
export disable_tqdm=True
export lr_scheduler=linear
export map_config=config/offset_map/closest_offset_en.yaml
export spot_noise=0.1
export asoc_noise=0.1
export other_ratio=0
export negative_ratio=1
export k_sparse=20
export use_ema=False
export use_sparsemax=False
export use_prompt=True
export use_ssi=True
export prompt_len=10
export prompt_dim=800
export init_prompt=True
export batch_size=16
export freeze_LM=False
export record2=0


export model_name="hf_models/t5-v1_1-base"
export data_name="relation/NYT"
export output_dir="output/relation_NYT"
export config="base_model_conf_nyt.ini"


# 检查命令行参数,要在其中
OPTS=$(getopt -o b:d:m:i:t:k:s:l:f:n:v --long batch:,device:,model:,data:,output:,k_sparse:,use_sparsemax:,use_ema:,add_null:,negative_ratio:,use_prompt:,prompt_len:,prompt_dim:,init_prompt:,freeze_LM:,record2:,config:,task:,run-time:,seed:,lr:,lr_scheduler:,label_smoothing:,epoch:,format:,eval_steps:,warmup_ratio:,constraint_decoding,verbose,preprocess,fp16:,negative_ratio:,random_prompt,max_source_length:,max_target_length:,spot_noise:,asoc_noise:,positive:,map_config:, -n 'parse-options' -- "$@")
if [ $? != 0 ]; then
  echo "Failed parsing options." >&2
  exit 1
fi
eval set -- "$OPTS"

# 循环取出test_finetune.bash命令行参数
while true; do
  case "$1" in
  -b | --batch)
    batch_size="$2"
    shift
    shift
    ;;
  -d | --device)
    CUDA_VISIBLE_DEVICES="$2"
    shift
    shift
    ;;
  -m | --model)
    model_name="$2"
    shift
    shift
    ;;
  -i | --data)
    data_name="$2"
    shift
    shift
    ;;
  --output)
    output_name="$2"
    shift
    shift
    ;;
  --k_sparse)
    k_sparse="$2"
    shift
    shift
    ;;
  --use_sparsemax)
    use_sparsemax="$2"
    shift
    shift
    ;;
  --use_ema)
    use_ema="$2"
    shift
    shift
    ;;
  --add_null)
    add_null="$2"
    shift
    shift
    ;;
  --negative_ratio)
    negative_ratio="$2"
    shift
    shift
    ;;
  --use_prompt)
    use_prompt="$2"
    shift
    shift
    ;;
  --prompt_dim)
    prompt_dim="$2"
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
  --config)
    config_name="$2"
    shift
    shift
    ;;
  -t | --task)
    task="$2"
    shift
    shift
    ;;
  -k | --run-time)
    run_time="$2"
    shift
    shift
    ;;
  -s | --seed)
    seed="$2"
    shift
    shift
    ;;
  -l | --lr)
    lr="$2"
    shift
    shift
    ;;
  -f | --format)
    decoding_format="$2"
    shift
    shift
    ;;
  -n | --negative_ratio)
    negative_ratio="$2"
    shift
    shift
    ;;
  -p | --positive)
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
  -v | --verbose)
    verbose=true
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

# 添加当前目录（一般是~/UIE）到Python Path（Python解释器寻找路径）
export PYTHONPATH="${PYTHONPATH}:./"
# 运行配置文件，配置文件中的变量都是环境变量
config="config/prompt_conf/${config_name}"
source ${config}

export lr=${LR_RATE}
export warmup_ratio=${WARMUP_PROP}
export label_smoothing=${LABEL_SMOOTHING}
export negative=${NEGATIVE}
export spot_noise=${NOISE}
export asoc_noise=${NOISE}
export task_name=${task_name}
export prompt_len=${prompt_len}
export max_source_length=${max_source_length}
export max_target_length=${max_target_length}


if [[ ${eval_steps} == 0 ]]
then
  evaluation_strategy='epoch'
else
  evaluation_strategy='steps'
fi
if [[ ${save_steps} == 0 ]]
then
  save_strategy='epoch'
else
  save_strategy='steps'
fi


data_folder=data/${data_name}
output_dir="output/${output_name}"

if [[ ${record2} == 0 ]]
then
  record2=${data_folder}/record.schema
else
  record2=${record2}
fi


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 run_prompt.py \
    --do_train --do_eval --do_predict ${constraint_decoding} ${fp16} \
    --use_fast_tokenizer=True \
    --from_checkpoint=True \
    --ddp_find_unused_parameters=False \
    --predict_with_generate \
    --evaluation_strategy=${evaluation_strategy} \
    --save_strategy=${save_strategy} \
    --metric_for_best_model eval_overall-F1 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --max_source_length=${max_source_length:-"256"} \
    --max_prefix_length=${max_prefix_length:-"-1"} \
    --max_target_length=${max_target_length:-"192"} \
    --num_train_epochs=${epoch} \
    --task=${task} \
    --train_file=${data_folder}/train.json \
    --validation_file=${data_folder}/val.json \
    --test_file=${data_folder}/test.json \
    --record_schema=${data_folder}/record.schema \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=$((batch_size * 4)) \
    --output_dir=${output_dir} \
    --overwrite_output_dir=False\
    --logging_dir=${output_dir}_log \
    --model_name_or_path=${model_name} \
    --learning_rate=${lr} \
    --source_prefix="${task}: " \
    --lr_scheduler_type=${lr_scheduler} \
    --label_smoothing_factor=${label_smoothing} \
    --eval_steps ${eval_steps} \
    --save_steps ${save_steps} \
    --decoding_format ${decoding_format} \
    --warmup_ratio ${warmup_ratio} \
    --preprocessing_num_workers=4 \
    --dataloader_num_workers=0 \
    --meta_positive_rate=${positive} \
    --skip_memory_metrics \
    --no_remove_unused_columns \
    --ordered_prompt=${ordered_prompt} \
    --save_better_checkpoint=False \
    --start_eval_step=${start_eval_step:-"0"} \
    --spot_noise=${spot_noise} \
    --asoc_noise=${asoc_noise} \
    --other_ratio=${other_ratio} \
    --k_sparse=${k_sparse} \
    --use_sparsemax=${use_sparsemax} \
    --use_ema=${use_ema} \
    --negative_ratio=${negative_ratio} \
    --task_name=${task_name} \
    --prompt_len=${prompt_len} \
    --prompt_dim=${prompt_dim} \
    --use_prompt=${use_prompt} \
    --use_ssi=${use_ssi} \
    --init_prompt=${init_prompt} \
    --freeze_LM=${freeze_LM} \
    --record2=${record2} \
    --seed=${seed} --disable_tqdm=${disable_tqdm}
