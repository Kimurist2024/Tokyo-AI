import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaEmppatheticTrainer:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        """モデルとトークナイザーを読み込み（4bit量子化）"""
        logger.info("Loading model and tokenizer...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = prepare_model_for_kbit_training(self.model)
        
    def setup_lora_config(self):
        """LoRA設定"""
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
    def load_dataset(self, train_file: str, val_file: str):
        """データセット読み込み"""
        logger.info("Loading datasets...")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            
        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
            
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        return train_dataset, val_dataset
        
    def format_prompt(self, example: Dict) -> str:
        """プロンプト作成"""
        system_prompt = "You are an empathetic assistant who provides supportive and understanding responses to help people feel heard and cared for."
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['output']}<|eot_id|>"""
        
        return prompt
        
    def train(self, train_dataset, val_dataset, output_dir: str = "./empathetic-llama3"):
        """ファインチューニング実行"""
        logger.info("Starting training...")
        
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            num_train_epochs=3,
            evaluation_strategy="steps",
            eval_steps=100,
            logging_steps=50,
            warmup_steps=100,
            logging_strategy="steps",
            learning_rate=2e-4,
            fp16=False,
            bf16=torch.cuda.is_bf16_supported(),
            group_by_length=True,
            report_to=None,
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=42
        )
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=None,
            formatting_func=self.format_prompt,
            tokenizer=self.tokenizer,
            args=training_arguments,
            max_seq_length=1024,
        )
        
        trainer.train()
        
        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed! Model saved to {output_dir}")
        
def main():
    trainer = LlamaEmppatheticTrainer()
    
    logger.info("Step 1: Loading model and tokenizer...")
    trainer.load_model_and_tokenizer()
    
    logger.info("Step 2: Setting up LoRA...")
    trainer.setup_lora_config()
    
    logger.info("Step 3: Loading datasets...")
    train_dataset, val_dataset = trainer.load_dataset(
        "empathetic_train.json",
        "empathetic_val.json"
    )
    
    logger.info("Step 4: Starting training...")
    trainer.train(train_dataset, val_dataset)
    
    logger.info("Training pipeline completed!")

if __name__ == "__main__":
    main()