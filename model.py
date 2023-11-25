from utils import yaml_read
from train import training_arguments
from transformers import AutoModelForMultipleChoice, Trainer
from transformers import EarlyStoppingCallback

config = yaml_read('config.yaml')

class ZAIC_Math:
    def __init__(self, config=config, use_pretrained=False, save=False): 
        self.config = config
        self.save = save
        if use_pretrained: 
            if self.config['model']['USE_PERT']:
                self.model = AutoModelForMultipleChoice.from_pretrained(MODEL)
                self.model = get_peft_model(model, peft_config)
                self.checkpoint = torch.load(f'{config['ckpt']}/pytorch_model.bin')
                self.model.load_state_dict(checkpoint)
            else:
                self.model = AutoModelForMultipleChoice.from_pretrained(f'{config['ckpt']}')

        else:
            self.model = AutoModelForMultipleChoice.from_pretrained(self.config['model']['ID'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['ID'])

        #  Set-up
        if self.config['model']['USE_PERT']:
            lora_conf = self.config['model']['LORA']
            print('We are using PEFT.')
            from peft import LoraConfig, get_peft_model, TaskType
            peft_config = LoraConfig(
                r=lora_conf['r'], lora_alpha=lora_conf['alpha'], task_type=TaskType.lora_conf['task_type'], 
                lora_dropout=lora_conf['lora_dropout'],
                bias=lora_conf['bias'], 
                inference_mode=lora_conf['inference_mode'],
                target_modules=lora_conf['target_modules'],
                modules_to_save=lora_conf['modules_to_save'],
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        if self.config['model']['FREEZE_EMBEDDINGS']:
            print('Freezing embeddings.')
            for param in self.model.deberta.embeddings.parameters():
                param.requires_grad = False
        if self.config['model']['FREEZE_LAYERS']>0:
            print(f'Freezing {FREEZE_LAYERS} layers.')
            for layer in self.model.deberta.encoder.layer[:FREEZE_LAYERS]:
                for param in layer.parameters():
                    param.requires_grad = False

    def map_at_3(self, predictions, labels):
        map_sum = 0
        pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
        for x,y in zip(pred,labels):
            z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
            map_sum += np.sum(z)
        return map_sum / len(predictions)

    def compute_metrics(self, p):
        predictions = p.predictions.tolist()
        labels = p.label_ids.tolist()
        return {"map@3": map_at_3(predictions, labels)}

    def train(self): 
        trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset_valid,
            compute_metrics = self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        trainer.train()
        if self.save: 
            trainer.save_model(f'./{config['ckpt']}/model_v{VER}')
            # torch.save(model.state_dict(), f'./{config['ckpt']}model_v{VER}/pytorch_model.bin')