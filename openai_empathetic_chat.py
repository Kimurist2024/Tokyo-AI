#!/usr/bin/env python3
"""
OpenAI GPT-4 Empathetic ChatBot with WandB Metrics
OpenAI GPTを使った共感的チャットボット
"""

import openai
import os
import time
import argparse
import wandb
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import json

# 環境変数の読み込み
load_dotenv()

class OpenAIEmpatheticChatBot:
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 use_wandb: bool = True,
                 project_name: str = "openai-empathetic-chatbot"):
        """
        OpenAI GPTを使った共感的チャットボットの初期化
        
        Args:
            model: 使用するOpenAIモデル
            use_wandb: WandBを使用するかどうか
            project_name: WandBプロジェクト名
        """
        # OpenAI APIキーの設定
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
        # WandBの初期化
        if use_wandb:
            self._init_wandb(project_name, model)
        self.use_wandb = use_wandb
        
        # システムプロンプトの定義
        self.system_prompt = """You are an empathetic AI assistant specialized in providing emotional support and understanding. Your responses should be:

1. **Warm and supportive**: Show genuine care and understanding
2. **Non-judgmental**: Accept all feelings without criticism
3. **Active listening**: Reflect back what the user is feeling
4. **Culturally sensitive**: Be aware of cultural differences in emotional expression
5. **Encouraging**: Offer hope and gentle guidance when appropriate
6. **Language adaptive**: Respond in the same language as the user's input

Guidelines:
- Use empathetic phrases like "I understand," "That sounds difficult," "Your feelings are valid"
- Ask gentle follow-up questions to show engagement
- Offer practical suggestions only when appropriate
- Maintain a caring, professional tone
- Keep responses concise but meaningful (50-200 words)

Remember: Your goal is to help people feel heard, understood, and supported."""
        
        # メトリクスの初期化
        self.conversation_metrics = {
            "response_times": [],
            "response_lengths": [],
            "token_usage": [],
            "user_inputs": [],
            "bot_responses": []
        }
        
        print(f"✅ OpenAI ChatBot initialized with model: {model}")
    
    def _init_wandb(self, project_name: str, model: str):
        """WandBの初期化"""
        try:
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(
                project=project_name,
                config={
                    "model": model,
                    "provider": "openai",
                    "framework": "openai",
                    "task": "empathetic_conversation"
                }
            )
            print("✅ WandB initialized successfully!")
        except Exception as e:
            print(f"⚠️  Could not initialize WandB: {e}")
            self.use_wandb = False
    
    def generate_response(self, 
                         user_input: str, 
                         max_tokens: int = 300,
                         temperature: float = 0.7) -> Tuple[str, Dict]:
        """
        GPTを使って共感的な応答を生成
        
        Args:
            user_input: ユーザーの入力
            max_tokens: 最大トークン数
            temperature: 温度パラメータ
            
        Returns:
            応答テキストとメトリクス
        """
        start_time = time.time()
        
        try:
            # GPT APIを呼び出し
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            # 応答とメタデータの取得
            bot_response = response.choices[0].message.content.strip()
            response_time = time.time() - start_time
            
            # トークン使用量
            usage = response.usage
            token_usage = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
            
            # メトリクスの計算
            metrics = {
                "response_time": response_time,
                "response_length": len(bot_response),
                "response_words": len(bot_response.split()),
                "input_length": len(user_input),
                "input_words": len(user_input.split()),
                **token_usage
            }
            
            # 共感度スコアの計算
            empathy_score = self._calculate_empathy_score(bot_response)
            metrics["empathy_score"] = empathy_score
            
            # メトリクスの保存
            self.conversation_metrics["response_times"].append(response_time)
            self.conversation_metrics["response_lengths"].append(len(bot_response))
            self.conversation_metrics["token_usage"].append(token_usage)
            self.conversation_metrics["user_inputs"].append(user_input)
            self.conversation_metrics["bot_responses"].append(bot_response)
            
            # WandBにログ
            if self.use_wandb:
                wandb.log(metrics)
            
            return bot_response, metrics
            
        except Exception as e:
            error_response = f"申し訳ありません。エラーが発生しました: {str(e)}"
            error_metrics = {
                "error": True,
                "response_time": time.time() - start_time,
                "response_length": len(error_response),
                "response_words": len(error_response.split())
            }
            return error_response, error_metrics
    
    def _calculate_empathy_score(self, response: str) -> float:
        """共感度スコアの計算（簡易版）"""
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
        print("🤖 OpenAI GPT Empathetic ChatBot")
        print(f"Model: {self.model}")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'metrics' to see conversation statistics")
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
                
                if not user_input.strip():
                    continue
                
                print("🤖 Bot: ", end="", flush=True)
                
                response, metrics = self.generate_response(user_input)
                print(response)
                
                # メトリクス表示
                print(f"\n📊 [{metrics['response_time']:.2f}s | "
                      f"{metrics.get('total_tokens', 'N/A')} tokens | "
                      f"Empathy: {metrics.get('empathy_score', 0):.2f}]")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _print_current_metrics(self):
        """現在のメトリクス表示"""
        if not self.conversation_metrics["response_times"]:
            print("📊 No metrics available yet.")
            return
        
        total_conversations = len(self.conversation_metrics["response_times"])
        avg_response_time = sum(self.conversation_metrics["response_times"]) / total_conversations
        avg_response_length = sum(self.conversation_metrics["response_lengths"]) / total_conversations
        
        total_tokens = sum(usage.get("total_tokens", 0) 
                          for usage in self.conversation_metrics["token_usage"])
        
        print("\n📊 Current Session Metrics:")
        print(f"  • Conversations: {total_conversations}")
        print(f"  • Avg Response Time: {avg_response_time:.2f}s")
        print(f"  • Avg Response Length: {avg_response_length:.0f} chars")
        print(f"  • Total Tokens Used: {total_tokens}")
        print()
    
    def _print_session_summary(self):
        """セッション終了時のサマリー"""
        if not self.conversation_metrics["response_times"]:
            return
        
        total_conversations = len(self.conversation_metrics["response_times"])
        total_time = sum(self.conversation_metrics["response_times"])
        total_tokens = sum(usage.get("total_tokens", 0) 
                          for usage in self.conversation_metrics["token_usage"])
        
        print("\n" + "="*50)
        print("📈 SESSION SUMMARY")
        print("="*50)
        print(f"Total Conversations: {total_conversations}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Total Tokens: {total_tokens}")
        print(f"Avg Time/Response: {total_time/total_conversations:.2f}s")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="OpenAI GPT Empathetic ChatBot")
    parser.add_argument("--model", default="gpt-4o-mini",
                       choices=["gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo", "o1-mini"],
                       help="OpenAI model to use")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable WandB logging")
    parser.add_argument("--project", default="openai-empathetic-chatbot",
                       help="WandB project name")
    parser.add_argument("--max_tokens", type=int, default=300,
                       help="Maximum tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for response generation")
    
    args = parser.parse_args()
    
    try:
        chatbot = OpenAIEmpatheticChatBot(
            model=args.model,
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
        print("Please check your OpenAI API key and try again.")

if __name__ == "__main__":
    main()