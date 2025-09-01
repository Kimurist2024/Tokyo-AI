"""
ローカル環境での実際のファインチューニング代替案
Groqの制限を回避してモデルを実際に学習させる方法
"""

import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

class LocalEmpatheticFinetune:
    """ローカル環境でのファインチューニング"""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """利用可能なオープンソースモデルを使用"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def setup_model(self):
        """モデルとトークナイザーのセットアップ"""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
    def prepare_dataset(self, data_file="empathetic_train.json"):
        """共感対話データセットの準備"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # データをトークナイズ
        tokenized_data = []
        for item in data:
            # 対話形式のテキストを作成
            text = f"User: {item['input']}\nAssistant: {item['output']}{self.tokenizer.eos_token}"
            
            # トークナイズ
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors="pt"
            )
            
            tokenized_data.append({
                'input_ids': tokens['input_ids'][0],
                'attention_mask': tokens['attention_mask'][0]
            })
        
        return Dataset.from_list(tokenized_data)
    
    def finetune(self, output_dir="./empathetic-model"):
        """実際のファインチューニング実行"""
        if not self.model or not self.tokenizer:
            self.setup_model()
            
        # データセット準備
        train_dataset = self.prepare_dataset()
        
        # 訓練設定
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            learning_rate=5e-5,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=True,
            report_to=None
        )
        
        # データコレーター
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # トレーナー
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        print("Starting fine-tuning...")
        trainer.train()
        
        # モデル保存
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")

def print_alternatives():
    """ファインチューニング代替手段の説明"""
    print("""
=== ファインチューニング代替手段 ===

1. 【Groq + プロンプトエンジニアリング】（現在実装済み）
   - システムプロンプトで共感的応答を誘導
   - 高速・安価・即座に利用可能
   - 実行: python groq_empathetic_chat.py

2. 【ローカルファインチューニング】
   - オープンソースモデルを実際に学習
   - 完全な制御とカスタマイズ
   - 実行: python local_finetune_alternative.py

3. 【OpenAI GPT-3.5/4 ファインチューニング】
   - $8-20程度で本格的ファインチューニング
   - 高品質な結果
   - OpenAI APIが必要

4. 【Hugging Face AutoTrain】
   - ブラウザベースの簡単ファインチューニング
   - 自動最適化
   - https://ui.autotrain.huggingface.co/

推奨: まずはGroq版で試して、必要に応じてローカル学習へ移行
    """)

def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--finetune":
        # 実際のローカルファインチューニング
        trainer = LocalEmpatheticFinetune()
        trainer.finetune()
    else:
        # 代替手段の説明
        print_alternatives()

if __name__ == "__main__":
    main()