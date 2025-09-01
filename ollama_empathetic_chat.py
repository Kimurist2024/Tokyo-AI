#!/usr/bin/env python3
"""
Ollama Empathetic ChatBot with WandB Metrics
Ollamaを使った共感的チャットボット（完全無料・ローカル）
"""

import requests
import json
import time
import argparse
import wandb
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# 環境変数の読み込み
load_dotenv()

class OllamaEmpatheticChatBot:
    def __init__(self, 
                 model: str = "llama3.2:3b",
                 base_url: str = "http://localhost:11434",
                 use_wandb: bool = True,
                 project_name: str = "ollama-empathetic-chatbot"):
        """
        Ollamaを使った共感的チャットボットの初期化
        
        Args:
            model: 使用するOllamaモデル
            base_url: OllamaサーバーのURL
            use_wandb: WandBを使用するかどうか
            project_name: WandBプロジェクト名
        """
        self.model = model
        self.base_url = base_url
        
        # WandBの初期化
        if use_wandb:
            self._init_wandb(project_name, model)
        self.use_wandb = use_wandb
        
        # システムプロンプト
        self.system_prompt = """You are an empathetic AI assistant who provides emotional support and understanding. Your responses should be:

1. Warm, caring, and supportive
2. Non-judgmental and accepting of all feelings
3. Focused on active listening and reflection
4. Culturally sensitive and appropriate
5. Encouraging when appropriate
6. Responsive in the same language as the user

Guidelines:
- Use empathetic language like "I understand", "That sounds difficult", "Your feelings are valid"
- Ask gentle follow-up questions to show engagement
- Keep responses concise but meaningful (2-4 sentences)
- Offer practical suggestions only when appropriate
- Always maintain a caring, professional tone

Remember: Your goal is to help people feel heard, understood, and emotionally supported."""
        
        # 会話履歴
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # メトリクス
        self.conversation_metrics = {
            "response_times": [],
            "response_lengths": [],
            "user_inputs": [],
            "bot_responses": []
        }
        
        # Ollamaの接続テスト
        self._test_connection()
        
        print(f"✅ Ollama ChatBot initialized with model: {model}")
    
    def _init_wandb(self, project_name: str, model: str):
        """WandBの初期化"""
        try:
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(
                project=project_name,
                config={
                    "model": model,
                    "provider": "ollama",
                    "framework": "ollama",
                    "task": "empathetic_conversation"
                }
            )
            print("✅ WandB initialized successfully!")
        except Exception as e:
            print(f"⚠️  Could not initialize WandB: {e}")
            self.use_wandb = False
    
    def _test_connection(self):
        """Ollamaサーバーへの接続テスト"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama server connection successful")
                
                # 利用可能モデルの確認
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model not in model_names:
                    print(f"⚠️  Model '{self.model}' not found. Available models: {model_names}")
                    print(f"To download: ollama pull {self.model}")
                else:
                    print(f"✅ Model '{self.model}' is available")
            else:
                print(f"❌ Ollama server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama server. Please run 'ollama serve'")
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
    
    def generate_response(self, user_input: str, temperature: float = 0.7) -> Tuple[str, Dict]:
        """
        Ollamaを使って共感的な応答を生成
        
        Args:
            user_input: ユーザーの入力
            temperature: 温度パラメータ
            
        Returns:
            応答テキストとメトリクス
        """
        start_time = time.time()
        
        # 会話履歴に追加
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Ollama APIを呼び出し
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": self.conversation_history,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                bot_response = result['message']['content'].strip()
                
                # 会話履歴に追加
                self.conversation_history.append({"role": "assistant", "content": bot_response})
                
                # 応答時間計算
                response_time = time.time() - start_time
                
                # メトリクス計算
                metrics = {
                    "response_time": response_time,
                    "response_length": len(bot_response),
                    "response_words": len(bot_response.split()),
                    "input_length": len(user_input),
                    "input_words": len(user_input.split())
                }
                
                # 共感度スコア
                empathy_score = self._calculate_empathy_score(bot_response)
                metrics["empathy_score"] = empathy_score
                
                # メトリクス保存
                self.conversation_metrics["response_times"].append(response_time)
                self.conversation_metrics["response_lengths"].append(len(bot_response))
                self.conversation_metrics["user_inputs"].append(user_input)
                self.conversation_metrics["bot_responses"].append(bot_response)
                
                # WandBにログ
                if self.use_wandb:
                    wandb.log(metrics)
                
                return bot_response, metrics
                
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                error_metrics = {
                    "error": True,
                    "response_time": time.time() - start_time,
                    "response_length": len(error_msg),
                    "response_words": len(error_msg.split())
                }
                return error_msg, error_metrics
                
        except requests.exceptions.ConnectionError:
            error_msg = "Ollamaサーバーに接続できません。'ollama serve'を実行してください。"
            error_metrics = {
                "error": True,
                "response_time": time.time() - start_time,
                "response_length": len(error_msg),
                "response_words": len(error_msg.split())
            }
            return error_msg, error_metrics
            
        except Exception as e:
            error_msg = f"エラー: {str(e)}"
            error_metrics = {
                "error": True,
                "response_time": time.time() - start_time,
                "response_length": len(error_msg),
                "response_words": len(error_msg.split())
            }
            return error_msg, error_metrics
    
    def _calculate_empathy_score(self, response: str) -> float:
        """共感度スコアの計算"""
        empathetic_keywords = [
            # 英語
            "understand", "feel", "sorry", "hear", "support", "care", "help",
            "difficult", "challenging", "valid", "important", "here for you",
            # 日本語
            "理解", "感じ", "つらい", "大変", "支援", "サポート", "聞く",
            "気持ち", "心配", "大丈夫", "頑張", "応援"
        ]
        
        response_lower = response.lower()
        matched_keywords = sum(1 for keyword in empathetic_keywords 
                             if keyword.lower() in response_lower)
        
        # 質問の存在もチェック
        has_question = '?' in response or '？' in response
        question_bonus = 0.1 if has_question else 0
        
        # 0-1のスケールで正規化
        empathy_score = min(1.0, (matched_keywords * 0.1) + question_bonus)
        return empathy_score
    
    def chat_loop(self):
        """対話ループ"""
        print("\n" + "="*70)
        print("🤖 Ollama Empathetic ChatBot (Complete Free & Local)")
        print(f"Model: {self.model}")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'metrics' to see conversation statistics")
        print("Type 'models' to see available models")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("💭 You: ")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for the conversation!")
                    self._print_session_summary()
                    break
                
                if user_input.lower() == 'metrics':
                    self._print_current_metrics()
                    continue
                    
                if user_input.lower() == 'models':
                    self._show_available_models()
                    continue
                
                if not user_input.strip():
                    continue
                
                print("🤖 Bot: ", end="", flush=True)
                
                response, metrics = self.generate_response(user_input)
                print(response)
                
                # メトリクス表示
                print(f"\n📊 [{metrics['response_time']:.2f}s | "
                      f"{metrics['response_words']} words | "
                      f"Empathy: {metrics.get('empathy_score', 0):.2f}]")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_available_models(self):
        """利用可能なモデル一覧表示"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print("\n📋 Available Ollama Models:")
                for model in models:
                    name = model['name']
                    size = model.get('size', 0)
                    size_mb = size / (1024*1024) if size > 0 else 0
                    print(f"  • {name} ({size_mb:.1f}MB)")
                print()
            else:
                print("❌ Could not retrieve model list")
        except Exception as e:
            print(f"❌ Error getting models: {e}")
    
    def _print_current_metrics(self):
        """現在のメトリクス表示"""
        if not self.conversation_metrics["response_times"]:
            print("📊 No metrics available yet.")
            return
        
        total_conversations = len(self.conversation_metrics["response_times"])
        avg_response_time = sum(self.conversation_metrics["response_times"]) / total_conversations
        avg_response_length = sum(self.conversation_metrics["response_lengths"]) / total_conversations
        
        print("\n📊 Current Session Metrics:")
        print(f"  • Conversations: {total_conversations}")
        print(f"  • Avg Response Time: {avg_response_time:.2f}s")
        print(f"  • Avg Response Length: {avg_response_length:.0f} chars")
        print()
    
    def _print_session_summary(self):
        """セッション終了時のサマリー"""
        if not self.conversation_metrics["response_times"]:
            return
        
        total_conversations = len(self.conversation_metrics["response_times"])
        total_time = sum(self.conversation_metrics["response_times"])
        
        print("\n" + "="*50)
        print("📈 SESSION SUMMARY")
        print("="*50)
        print(f"Total Conversations: {total_conversations}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Avg Time/Response: {total_time/total_conversations:.2f}s")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Ollama Empathetic ChatBot")
    parser.add_argument("--model", default="llama3.2:3b",
                       help="Ollama model to use (e.g., llama3.2:3b, llama3.1:8b, elyza:jp-8b)")
    parser.add_argument("--url", default="http://localhost:11434",
                       help="Ollama server URL")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable WandB logging")
    parser.add_argument("--project", default="ollama-empathetic-chatbot",
                       help="WandB project name")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for response generation")
    
    args = parser.parse_args()
    
    try:
        chatbot = OllamaEmpatheticChatBot(
            model=args.model,
            base_url=args.url,
            use_wandb=not args.no_wandb,
            project_name=args.project
        )
        
        # 対話開始
        chatbot.chat_loop()
        
        # WandB終了
        if chatbot.use_wandb:
            wandb.finish()
            
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        print("Please ensure Ollama is running with 'ollama serve'")

if __name__ == "__main__":
    main()