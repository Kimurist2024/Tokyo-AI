#!/usr/bin/env python3
"""
Advanced Empathetic ChatBot with WandB Metrics Tracking
高精度モデルとメトリクス追跡機能を備えた共感的チャットボット
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import wandb
import os
from dotenv import load_dotenv
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

# 環境変数の読み込み
load_dotenv()

# 高精度モデルの選択肢
HIGH_ACCURACY_MODELS = {
    # 日本語対応の高精度モデル
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",  # Llama 2 7B (要アクセス許可)
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",  # Llama 3 8B (要アクセス許可)
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",  # Mistral 7B
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Mixtral 8x7B (MoE)
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",  # Qwen2 7B (多言語対応)
    "gemma-7b": "google/gemma-7b-it",  # Google Gemma 7B
    "phi3-medium": "microsoft/Phi-3-medium-4k-instruct",  # Phi-3 Medium (14B)
    "yi-6b": "01-ai/Yi-6B-Chat",  # Yi 6B (中国語・英語)
    "japanese-stablelm": "stabilityai/japanese-stablelm-instruct-gamma-7b",  # 日本語特化
    "calm2-7b": "cyberagent/calm2-7b-chat",  # CALM2 7B (日本語)
}

class AdvancedEmpatheticChatBot:
    def __init__(self, 
                 model_name: str = "mistral-7b", 
                 adapter_path: str = None,
                 use_wandb: bool = True,
                 project_name: str = "empathetic-chatbot"):
        """
        高精度な共感的チャットボットの初期化
        
        Args:
            model_name: モデル名のキー（HIGH_ACCURACY_MODELS内）
            adapter_path: LoRAアダプターのパス（オプション）
            use_wandb: WandBを使用するかどうか
            project_name: WandBプロジェクト名
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # WandBの初期化
        if use_wandb:
            self._init_wandb(project_name, model_name)
        self.use_wandb = use_wandb
        
        # モデルパスの取得
        if model_name in HIGH_ACCURACY_MODELS:
            base_model_path = HIGH_ACCURACY_MODELS[model_name]
            print(f"Selected model: {model_name} ({base_model_path})")
        else:
            base_model_path = model_name
            print(f"Using custom model: {base_model_path}")
        
        # トークナイザーの読み込み
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # モデルの読み込み
        print("Loading base model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to Phi-3 mini model...")
            base_model_path = "microsoft/Phi-3-mini-4k-instruct"
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True
            )
        
        # LoRAアダプターの読み込み
        if adapter_path:
            print(f"Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32
            )
        
        self.model.eval()
        self.model_name = model_name
        print("Model loaded successfully!")
        
        # メトリクスの初期化
        self.conversation_metrics = {
            "response_times": [],
            "response_lengths": [],
            "user_inputs": [],
            "bot_responses": []
        }
    
    def _init_wandb(self, project_name: str, model_name: str):
        """WandBの初期化"""
        try:
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(
                project=project_name,
                config={
                    "model": model_name,
                    "device": str(torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"),
                    "framework": "transformers",
                    "task": "empathetic_conversation"
                }
            )
            print("WandB initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not initialize WandB: {e}")
            self.use_wandb = False
    
    def create_prompt(self, user_input: str) -> str:
        """モデルに応じた共感的なプロンプトを作成"""
        system_prompt = """You are an empathetic AI assistant specialized in providing emotional support. 
Your responses should be:
1. Warm, understanding, and supportive
2. Non-judgmental and accepting
3. Focused on acknowledging feelings
4. Offering gentle encouragement when appropriate
5. Culturally sensitive and aware

Please respond in the same language as the user's input."""
        
        # モデルタイプに応じたプロンプト形式
        if "llama" in self.model_name.lower():
            prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_input} [/INST]"""
        elif "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower():
            prompt = f"""<s>[INST] {system_prompt}

{user_input} [/INST]"""
        elif "phi" in self.model_name.lower():
            prompt = f"""<|system|>
{system_prompt}<|end|>
<|user|>
{user_input}<|end|>
<|assistant|>"""
        elif "qwen" in self.model_name.lower():
            prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
        else:
            # デフォルトフォーマット
            prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
        
        return prompt
    
    def calculate_metrics(self, user_input: str, response: str, response_time: float) -> Dict:
        """応答のメトリクスを計算"""
        metrics = {
            "response_time": response_time,
            "response_length": len(response),
            "response_words": len(response.split()),
            "input_length": len(user_input),
            "input_words": len(user_input.split()),
            "tokens_per_second": len(response.split()) / response_time if response_time > 0 else 0
        }
        
        # 感情的な言葉の検出（簡易版）
        empathetic_keywords = [
            "understand", "feel", "sorry", "hear", "support", "care", "help",
            "理解", "感じ", "つらい", "大変", "支援", "サポート", "聞く"
        ]
        empathy_score = sum(1 for keyword in empathetic_keywords if keyword in response.lower())
        metrics["empathy_score"] = empathy_score
        
        return metrics
    
    def generate_response(self, 
                         user_input: str, 
                         max_length: int = 512, 
                         temperature: float = 0.7,
                         top_p: float = 0.95,
                         repetition_penalty: float = 1.1) -> Tuple[str, Dict]:
        """応答を生成しメトリクスを記録"""
        start_time = time.time()
        
        prompt = self.create_prompt(user_input)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True if "phi" not in self.model_name.lower() else False
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 応答の抽出
        response = self._extract_response(full_response, prompt, user_input)
        
        # 応答時間の計算
        response_time = time.time() - start_time
        
        # メトリクスの計算
        metrics = self.calculate_metrics(user_input, response, response_time)
        
        # メトリクスの保存
        self.conversation_metrics["response_times"].append(response_time)
        self.conversation_metrics["response_lengths"].append(len(response))
        self.conversation_metrics["user_inputs"].append(user_input)
        self.conversation_metrics["bot_responses"].append(response)
        
        # WandBにログ
        if self.use_wandb:
            wandb.log(metrics)
        
        return response, metrics
    
    def _extract_response(self, full_response: str, prompt: str, user_input: str) -> str:
        """モデルの出力から応答部分を抽出"""
        # 各種終了タグで分割
        for end_tag in ["[/INST]", "<|assistant|>", "<|im_start|>assistant", "Assistant:"]:
            if end_tag in full_response:
                response = full_response.split(end_tag)[-1]
                break
        else:
            response = full_response
        
        # 特殊トークンの削除
        special_tokens = [
            "<|end|>", "<|endoftext|>", "<|im_end|>", "</s>", "<s>",
            "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
            "<|system|>", "<|user|>", "<|assistant|>"
        ]
        for token in special_tokens:
            response = response.replace(token, "")
        
        # プロンプトとユーザー入力の削除
        response = response.replace(prompt, "").replace(user_input, "")
        
        # システムプロンプトの削除
        if "empathetic AI assistant" in response:
            lines = response.split("\n")
            response = "\n".join([line for line in lines if "empathetic" not in line.lower()])
        
        return response.strip()
    
    def evaluate_conversation(self) -> Dict:
        """会話全体の評価メトリクスを計算"""
        if not self.conversation_metrics["response_times"]:
            return {}
        
        evaluation = {
            "total_conversations": len(self.conversation_metrics["response_times"]),
            "avg_response_time": np.mean(self.conversation_metrics["response_times"]),
            "avg_response_length": np.mean(self.conversation_metrics["response_lengths"]),
            "total_time": sum(self.conversation_metrics["response_times"]),
            "min_response_time": min(self.conversation_metrics["response_times"]),
            "max_response_time": max(self.conversation_metrics["response_times"])
        }
        
        if self.use_wandb:
            wandb.log({"evaluation": evaluation})
        
        return evaluation
    
    def chat_loop(self):
        """対話ループ"""
        print("\n" + "="*60)
        print("🤖 Advanced Empathetic ChatBot with Metrics")
        print(f"Model: {self.model_name}")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'metrics' to see conversation statistics")
        print("="*60 + "\n")
        
        while True:
            user_input = input("あなた: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nチャットを終了します。")
                evaluation = self.evaluate_conversation()
                if evaluation:
                    print("\n📊 Session Statistics:")
                    for key, value in evaluation.items():
                        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
                break
            
            if user_input.lower() == 'metrics':
                evaluation = self.evaluate_conversation()
                if evaluation:
                    print("\n📊 Current Statistics:")
                    for key, value in evaluation.items():
                        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
                else:
                    print("No metrics available yet.")
                continue
            
            if not user_input.strip():
                continue
            
            print("ChatBot: ", end="", flush=True)
            
            try:
                response, metrics = self.generate_response(user_input)
                print(response)
                print(f"\n⏱ Response time: {metrics['response_time']:.2f}s | 📝 Words: {metrics['response_words']}")
            except Exception as e:
                print(f"エラーが発生しました: {e}")
            
            print()

def main():
    parser = argparse.ArgumentParser(description="Advanced Empathetic ChatBot with Metrics")
    parser.add_argument("--model", default="mistral-7b", 
                       choices=list(HIGH_ACCURACY_MODELS.keys()) + ["custom"],
                       help="Model to use")
    parser.add_argument("--custom_model", default=None,
                       help="Custom model path if --model is 'custom'")
    parser.add_argument("--adapter_path", default=None, 
                       help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable WandB logging")
    parser.add_argument("--project", default="empathetic-chatbot",
                       help="WandB project name")
    
    args = parser.parse_args()
    
    model_name = args.custom_model if args.model == "custom" else args.model
    
    chatbot = AdvancedEmpatheticChatBot(
        model_name=model_name,
        adapter_path=args.adapter_path,
        use_wandb=not args.no_wandb,
        project_name=args.project
    )
    
    chatbot.chat_loop()
    
    # WandBの終了
    if chatbot.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()