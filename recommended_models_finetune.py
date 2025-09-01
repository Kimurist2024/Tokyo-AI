"""
推奨モデルでのファインチューニング実装
用途とリソースに応じたモデル選択ガイド
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import json

class RecommendedModelFinetune:
    
    # 推奨モデルリスト（リソース別）
    MODELS = {
        "lightweight": {
            "microsoft/Phi-3-mini-4k-instruct": {
                "params": "3.8B",
                "vram": "8GB",
                "features": ["超軽量", "高速", "エッジデバイス対応"]
            },
            "google/gemma-2-9b-it": {
                "params": "9B", 
                "vram": "12GB",
                "features": ["効率的", "Googleの最新技術"]
            }
        },
        "balanced": {
            "Qwen/Qwen2.5-7B-Instruct": {
                "params": "7B",
                "vram": "16GB", 
                "features": ["日本語優秀", "コード生成", "バランス良い"]
            },
            "mistralai/Mistral-7B-Instruct-v0.2": {
                "params": "7B",
                "vram": "16GB",
                "features": ["対話特化", "Apache2.0", "商用利用可"]
            }
        },
        "high_performance": {
            "unsloth/llama-3.1-8b-instruct-bnb-4bit": {
                "params": "8B (4bit)",
                "vram": "12GB",
                "features": ["最高性能", "量子化済み", "日本語対応"]
            },
            "microsoft/DialoGPT-large": {
                "params": "774M",
                "vram": "6GB", 
                "features": ["対話特化", "軽量", "すぐ使える"]
            }
        },
        "japanese_specialized": {
            "cyberagent/calm2-7b-chat": {
                "params": "7B",
                "vram": "16GB",
                "features": ["日本製", "日本語最適化"]
            },
            "stabilityai/japanese-stablelm-instruct-alpha-7b": {
                "params": "7B", 
                "vram": "16GB",
                "features": ["日本語特化", "商用利用可"]
            }
        }
    }
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def recommend_model(self, gpu_vram_gb=16, use_case="general"):
        """リソースと用途に基づくモデル推奨"""
        print(f"\n=== GPU VRAM: {gpu_vram_gb}GB, 用途: {use_case} ===")
        
        recommendations = []
        
        for category, models in self.MODELS.items():
            for model_name, specs in models.items():
                required_vram = int(specs["vram"].replace("GB", ""))
                
                if required_vram <= gpu_vram_gb:
                    recommendations.append({
                        "name": model_name,
                        "category": category,
                        "params": specs["params"],
                        "vram": specs["vram"],
                        "features": specs["features"]
                    })
        
        # おすすめ順にソート
        recommendations.sort(key=lambda x: int(x["vram"].replace("GB", "")), reverse=True)
        
        print("\n推奨モデル:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"{i}. {rec['name']}")
            print(f"   パラメータ: {rec['params']}, VRAM: {rec['vram']}")
            print(f"   特徴: {', '.join(rec['features'])}")
            print()
            
        return recommendations[0]["name"] if recommendations else None
    
    def setup_model_with_quantization(self):
        """4bit量子化でモデルをロード"""
        print(f"Loading model with 4-bit quantization: {self.model_name}")
        
        # 4bit量子化設定
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # トークナイザー
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # モデル
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # LoRA設定
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"Model loaded successfully!")
        self.model.print_trainable_parameters()
    
    def finetune_empathetic(self, train_file="empathetic_train.json"):
        """共感的応答のファインチューニング"""
        if not self.model:
            self.setup_model_with_quantization()
        
        # データセット読み込み
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # プロンプトテンプレート
        def format_prompt(example):
            return f"""### 指示:
以下のメッセージに共感的に応答してください。

### ユーザー:
{example['input']}

### アシスタント:
{example['output']}"""
        
        # 訓練設定
        training_args = TrainingArguments(
            output_dir=f"./empathetic-{self.model_name.split('/')[-1]}",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=200,
            optim="paged_adamw_32bit",
            warmup_steps=50,
            group_by_length=True,
            fp16=False,
            bf16=torch.cuda.is_bf16_supported(),
            report_to=None
        )
        
        # トレーナー
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_data,
            formatting_func=format_prompt,
            tokenizer=self.tokenizer,
            args=training_args,
            max_seq_length=1024,
            packing=False
        )
        
        print("Starting fine-tuning...")
        trainer.train()
        
        # 保存
        output_dir = training_args.output_dir
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")
        
        return output_dir

def main():
    """メイン実行"""
    import sys
    
    # GPU情報確認
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_memory:.1f}GB")
        vram_gb = int(gpu_memory)
    else:
        print("CUDA not available. Using CPU (not recommended for fine-tuning)")
        vram_gb = 0
    
    trainer = RecommendedModelFinetune()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--recommend":
        # モデル推奨
        recommended = trainer.recommend_model(vram_gb)
        if recommended:
            print(f"\n最推奨モデル: {recommended}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--finetune":
        # 実際のファインチューニング
        model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-7B-Instruct"
        trainer.model_name = model_name
        trainer.finetune_empathetic()
    else:
        # 情報表示
        trainer.recommend_model(vram_gb)
        print("\n使用方法:")
        print("python recommended_models_finetune.py --recommend  # モデル推奨")
        print("python recommended_models_finetune.py --finetune [model_name]  # 実行")

if __name__ == "__main__":
    main()