#!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
import numpy as np
from datasets import load_dataset
import random


import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    EarlyStoppingCallback,
    default_data_collator,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from uie.extraction import constants
from uie.extraction.record_schema import RecordSchema
from uie.extraction.extraction_metrics import get_extract_metrics
from uie.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser
from uie.extraction.dataset_processer import PrefixGenerator
from uie.seq2seq.constrained_seq2seq import ConstraintSeq2SeqTrainingArguments, EMA
from uie.seq2seq.constrained_seq2seq_prompt import (
    ConstraintSeq2SeqPromptTrainer, 
    ConstraintSeq2SeqPromptSparseTrainer
)

from uie.seq2seq.data_collator import (
    PromptForMetaSeq2Seq,
    PromptSSIGenerator,
)
from uie.seq2seq.features import RecordFeature
from uie.seq2seq.t5_bert_tokenizer import T5BertTokenizer
from uie.seq2seq.trainer_arguments import ModelArguments, DataTrainingArguments, PromptArguments
from uie.seq2seq.models import T5Prompt 

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

def get_negative_samples(l, k):
    from thefuzz import fuzz
    from gensim.test.utils import datapath, get_tmpfile
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove_file = datapath('/zjunlp/ghh/.cache/GloVe/glove.6B.300d.txt')
    word2vec_glove_file = get_tmpfile("glove.6B.300d.word2vec.txt")
    glove2word2vec(glove_file, word2vec_glove_file)
    model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

    negative_l = []
    for i in l:
        try:
            sim = model.most_similar(i.split()[0])
        except KeyError:
            continue
        cnt = 10
        for (x, _) in sim:
            if fuzz.ratio(i, x) < 65:
                if cnt > 0 and x not in l and x not in negative_l:
                    negative_l.append(x)
                    cnt -=1 

    return random.sample(negative_l, k)



def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConstraintSeq2SeqTrainingArguments, PromptArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, prompt_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, prompt_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    log_name = training_args.output_dir.split('/')[-1]
    os.makedirs(training_args.logging_dir, exist_ok = True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(training_args.logging_dir+f'/{log_name}.txt', mode = 'w', encoding = 'utf-8')],
    )
    logger.setLevel(logging.INFO)

    logger.info(f"last_checkpoint: {last_checkpoint}")
    logger.info("Options:")
    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)
    logger.info(prompt_args)

    cwd = os.getcwd()
    logger.info(cwd)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f", distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    seed_torch(training_args.seed)


    if data_args.dataset_name is not None:
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
    logger.info(data_files)  # {'train': 'data/text2spotasoc/absa/14lap/train.json',...
    datasets = load_dataset("uie_json.py", data_files=data_files)
    
    logger.info(datasets)   # train: Dataset({features: ['text', 'tokens', 'record', 'entity_offsets', 'relation_offsets', 'event_offsets', 'spot', 'asoc', 'spot_asoc', 'entity', 'relation', 'event'],num_rows: 906})
    logger.info("Load Config: %s" % model_args.config_name if model_args.config_name else model_args.model_name_or_path)

    config = AutoConfig.from_pretrained(
        f'{cwd}/hf_models/mix',
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.max_length = data_args.max_target_length


    if 'char' in model_args.model_name_or_path:
        tokenizer = T5BertTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )


    to_remove_token_list = list()     # 在postprocess_text中有用到
    if tokenizer.bos_token:
        to_remove_token_list += [tokenizer.bos_token]
    if tokenizer.eos_token:
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:
        to_remove_token_list += [tokenizer.pad_token]
    logger.info(f"tokenizer.bos_token: {tokenizer.bos_token}, tokenizer.bos_token_id: {tokenizer.bos_token_id}")
    logger.info(f"tokenizer.eos_token: {tokenizer.eos_token}, tokenizer.eos_token_id: {tokenizer.eos_token_id}")
    logger.info(f"tokenizer.pad_token: {tokenizer.pad_token}, tokenizer.pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"Tokenizer Length: {len(tokenizer)}")
    logger.info(f"Padding side: {tokenizer.padding_side}")


    if training_args.do_train:
        to_add_special_token = list()
        for special_token in [constants.type_start, constants.type_end, constants.text_start, constants.span_start, constants.spot_prompt, constants.asoc_prompt]:
            if special_token not in tokenizer.get_vocab():
                to_add_special_token += [special_token]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": tokenizer.special_tokens_map_extended['additional_special_tokens'] + to_add_special_token}
        )


    if 'prompt' in model_args.model_name_or_path:
        model = T5Prompt(tokenizer, f'{cwd}/hf_models/mix', prompt_args)
        model.load_state_dict(torch.load(os.path.join(model_args.model_name_or_path, 'pytorch_model.bin')))
    else:
        model = T5Prompt(           # define models
            tokenizer,
            model_args.model_name_or_path,
            prompt_args,
        )
    logger.info(f"Tokenizer Length: {len(tokenizer)}")   

    ema = None
    if training_args.use_ema:
        ema = EMA(model, 0.99, training_args.device)
        ema.register()


    if data_args.record_schema and os.path.exists(data_args.record_schema):
        record_schema = RecordSchema.read_from_file(data_args.record_schema)
    else:
        record_schema = None


    if prompt_args.init_prompt:
        logger.info(f"init_prompt? {prompt_args.init_prompt}")

        spot_prompt_id = tokenizer.encode(constants.spot_prompt, add_special_tokens = False)
        asoc_prompt_id = tokenizer.encode(constants.asoc_prompt, add_special_tokens = False)

        negative_file = '/'.join(data_args.train_file.split('/')[:-1]) + '/negative.pt'
        if os.path.exists(negative_file):
            logger.info(f"Load from {negative_file}")
            ng = torch.load(negative_file)
            negative_sample = ng["negative_sample"]
            negative_sample_ids = ng["negative_sample_ids"]
            spot_ids = ng["spot_ids"]
            asoc_ids = ng["asoc_ids"]
        else:
            spot_ids = []
            asoc_ids = []
            record_schema2 = RecordSchema.read_from_file(prompt_args.record2)
            for spot in record_schema2.type_list:
                spot_ids.append(tokenizer.encode(spot, add_special_tokens = False))     
            for asoc in record_schema2.role_list:
                asoc_ids.append(tokenizer.encode(asoc, add_special_tokens = False))
            neg_len = prompt_args.prompt_len - len(record_schema2.type_list) - len(record_schema2.role_list) - 5
            if data_args.task_name == 'relation':
                negative_sample = get_negative_samples(record_schema2.role_list, neg_len)
            else:
                negative_sample = get_negative_samples(record_schema2.type_list, neg_len)
            negative_sample_ids = []
            for it in negative_sample:
                negative_sample_ids.append(tokenizer.encode(it, add_special_tokens = False))
            ng = {"negative_sample": negative_sample, "negative_sample_ids": negative_sample_ids, "spot_ids": spot_ids, "asoc_ids": asoc_ids}
            torch.save(ng, negative_file)
            logger.info(f"Save to {negative_file}")

        logger.info(f"spot_ids: {spot_ids}")
        logger.info(f"asoc_ids: {asoc_ids}")
        logger.info(f"data_args.task_name: {data_args.task_name}")
        logger.info(f"negative_sample: {negative_sample}")
        logger.info(f"negative_sample_ids: {negative_sample_ids}")

        model.init_prompt(spot_ids, asoc_ids, negative_sample_ids, spot_prompt_id, asoc_prompt_id, [tokenizer.pad_token_id])



    if data_args.source_prefix is not None:
        if data_args.source_prefix == 'schema':
            prefix = PrefixGenerator.get_schema_prefix(schema=record_schema)
        elif data_args.source_prefix.startswith('meta'):
            prefix = ""
        else:
            prefix = data_args.source_prefix
    else:
        prefix = ""
    logger.info(f"Prefix: {prefix}")
    logger.info(f"Prefix Length: {len(tokenizer.tokenize(prefix))}")


    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return


    text_column = data_args.text_column
    record_column = data_args.record_column
    logger.info('Using src: %s and tgt: %s' % (text_column, record_column)) # Using src: text and tgt: record

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.error(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[record_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(_label if _label != tokenizer.pad_token_id else -100) for _label in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        model_inputs['sample_prompt'] = [False] * len(model_inputs['input_ids'])
        if data_args.source_prefix is not None and data_args.source_prefix.startswith('meta'):
            model_inputs['spots'] = examples['spot']
            model_inputs['asocs'] = examples['asoc']
            model_inputs['spot_asoc'] = examples['spot_asoc']
            # sample_prompt=True for Finetune and Pretrain
            model_inputs['sample_prompt'] = [True] * len(model_inputs['input_ids'])
        return model_inputs

    def preprocess_function_eval(examples):
        model_inputs = preprocess_function(examples)
        # sample_prompt=False for evaluation
        model_inputs['sample_prompt'] = [False] * len(model_inputs['input_ids'])
        return model_inputs

    def postprocess_text(x_str):
        # Clean `bos` `eos` `pad` for cleaned text
        for to_remove_token in to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()

    logger.info("Start Data Preprocessing ...")

    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            features=RecordFeature,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            features=RecordFeature,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            features=RecordFeature,
        )

    logger.info("End Data Preprocessing ...")

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:     # If False, will pad the samples dynamically when batching to the maximum length in the batch
        data_collator = default_data_collator
    elif data_args.source_prefix.startswith('meta'):

        if data_args.spot_noise > 0 or data_args.asoc_noise > 0:
            if data_args.decoding_format == 'spotasoc':
                spot_asoc_nosier = SpotAsocNoiser(
                    spot_noise_ratio=data_args.spot_noise,
                    asoc_noise_ratio=data_args.asoc_noise,
                    null_span=constants.null_span,
                )
            else:
                raise NotImplementedError(
                    f"decoding_format {data_args.decoding_format} is not implemented."
                )
        else:
            spot_asoc_nosier = None


        if data_args.task_name == 'relation':
            spot_negative = data_args.meta_negative
            asoc_negative = int(len(record_schema.role_list) * data_args.negative_ratio)
        else:
            spot_negative = int(len(record_schema.type_list) * data_args.negative_ratio) 
            asoc_negative = data_args.meta_negative
        logger.info(f"len(record_schema.type_list): {len(record_schema.type_list)}")
        logger.info(f"len(record_schema.role_list): {len(record_schema.role_list)}")   
        logger.info(f"data_args.negative_ratio: {data_args.negative_ratio}")
        logger.info(f"data_args.meta_negative: {data_args.meta_negative}")
        logger.info(f"task name: {data_args.task_name}")   
        logger.info(f"spot_negative: {spot_negative}")
        logger.info(f"asoc_negative: {asoc_negative}")


        data_collator = PromptForMetaSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            max_length=data_args.max_source_length,
            max_prefix_length=data_args.max_prefix_length,
            max_target_length=data_args.max_target_length,
            negative_sampler=PromptSSIGenerator(
                tokenizer=tokenizer,
                schema=record_schema,
                negative_list=negative_sample,      # 上面的负样本
                positive_rate=data_args.meta_positive_rate,
                spot_negative=spot_negative,
                asoc_negative=asoc_negative,
                other_ratio=data_args.other_ratio,
                ordered_prompt=data_args.ordered_prompt,
                task_name=data_args.task_name, 
            ),
            spot_asoc_nosier=spot_asoc_nosier,
            decoding_format=data_args.decoding_format,
            use_ssi=prompt_args.use_ssi,
        )


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_preds = [postprocess_text(x) for x in decoded_preds]
        decoded_labels = [postprocess_text(x) for x in decoded_labels]

        result = get_extract_metrics(
            pred_lns=decoded_preds,
            tgt_lns=decoded_labels,
            label_constraint=record_schema,
            decoding_format=data_args.decoding_format,
        )

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    logger.info(f"use_ssi? {prompt_args.use_ssi}")
    logger.info(f"use_sparsemax? {training_args.use_sparsemax}")
    logger.info(f"use_ema? {training_args.use_ema}")
    logger.info(f"ema? {ema}")
    train_dict = {"ConstraintSeq2SeqPromptTrainer": ConstraintSeq2SeqPromptTrainer, "ConstraintSeq2SeqPromptSparseTrainer": ConstraintSeq2SeqPromptSparseTrainer}
    s_sparsemax = "ConstraintSeq2SeqPromptTrainer"
    if training_args.use_sparsemax:
        s_sparsemax = "ConstraintSeq2SeqPromptSparseTrainer"
    #early_stop = EarlyStoppingCallback(early_stopping_patience = 5)
    #logger.info(f"EarlyStoppingCallback: {early_stop}")
    trainer = train_dict[s_sparsemax](
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        ema=ema,
        decoding_type_schema=record_schema,
        decoding_format=data_args.decoding_format,
        source_prefix=prefix,
        #callbacks=[early_stop, ],
        task=data_args.task_name,
    )
    
    # Training
    checkpoint = None
    if training_args.do_train:
        if model_args.from_checkpoint:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint

        logger.info(f"checkpoint: {checkpoint}")
        print(tokenizer.bos_token_id)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        # load_best_model_at_end=True，会在训练结束后加载最佳模型，然后save_model到output_dir

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate()
        results = {k: round(v, 4) for k, v in results.items()}    # 写到 "eval_results_seq2seq.txt"

        eval_results = trainer.predict(
            eval_dataset,
            metric_key_prefix="eval",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )        # 写到 "eval_preds_seq2seq.txt"

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            if training_args.predict_with_generate:
                eval_preds = tokenizer.batch_decode(
                    eval_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                eval_preds = [postprocess_text(pred) for pred in eval_preds]
                output_test_preds_file = os.path.join(training_args.output_dir, "eval_preds_seq2seq.txt")   
                # 只生成了preds_seq2seq.txt，没有生成preds_record.txt
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(eval_preds))

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        test_metrics = test_results.metrics
        test_metrics["test_loss"] = round(test_metrics["test_loss"], 4)

        output_test_result_file = os.path.join(training_args.output_dir, "test_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_test_result_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in sorted(test_metrics.items()):
                    logger.info(f"{key} = {value}")
                    writer.write(f"{key} = {value}\n")

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                test_preds = [postprocess_text(pred) for pred in test_preds]
                output_test_preds_file = os.path.join(training_args.output_dir, "test_preds_seq2seq.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))

    return results
  



if __name__ == "__main__":
    main()
