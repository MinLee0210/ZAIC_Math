from utils import yaml_read
from transformers import TrainingArguments

config = yaml_read('config.yaml')
train_conf = config['train']

training_args = TrainingArguments(
    warmup_ratio=train_conf['warmup_ratio'],
    learning_rate=train_conf['learning_rate'],
    per_device_train_batch_size=train_conf['per_device_train_batch_size'],
    per_device_eval_batch_size=train_conf['per_device_eval_batch_size'],
    num_train_epochs=train_conf['num_train_epochs'],
    report_to=train_conf['report_to'],
    output_dir = train_conf['output_dir'],
    overwrite_output_dir=train_conf['overwrite_output_dir'],
    fp16=train_conf['fp16'],
    gradient_accumulation_steps=train_conf['gradient_accumulation_steps'],
    logging_steps=train_conf['logging_steps'],
    evaluation_strategy=train_conf['evaluation_strategy'],
    eval_steps=train_conf['eval_steps'],
    save_strategy=train_conf['save_strategy'],
    save_steps=train_conf['save_steps'],
    load_best_model_at_end=train_conf['load_best_model_at_end'],
    metric_for_best_model=train_conf['metric_for_best_model'],
    lr_scheduler_type=train_conf['lr_scheduler_type'],
    weight_decay=train_conf['weight_decay'],
    save_total_limit=train_conf['save_total_limit'],
)
