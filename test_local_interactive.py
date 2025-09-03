"""
ローカルモデルとの対話テスト（OpenAI API不要、ユーザー入力式）
"""

import requests
import json
from typing import List, Dict

class LocalModelChat:
    def __init__(self, base_url="http://localhost:11434"):
        """Ollamaローカルサーバーとの接続を初期化"""
        self.base_url = base_url
        self.model_name = "llama3.2:3b"  # 軽量な日本語対応モデル
    
    def is_ollama_running(self) -> bool:
        """Ollamaサーバーが実行中かチェック"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def chat_with_local_model(self, messages: List[Dict[str, str]]) -> str:
        """ローカルモデルとチャット"""
        
        if not self.is_ollama_running():
            return "❌ Ollamaサーバーが起動していません。\n起動方法:\n1. ollama serve\n2. ollama pull llama3.2:3b"
        
        # メッセージを1つのプロンプトに結合
        prompt = self.format_messages_for_ollama(messages)
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "応答がありませんでした")
            else:
                return f"❌ エラー: {response.status_code}"
                
        except Exception as e:
            return f"❌ 接続エラー: {e}"
    
    def format_messages_for_ollama(self, messages: List[Dict[str, str]]) -> str:
        """OpenAI形式のメッセージをOllama用プロンプトに変換"""
        
        formatted_prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_prompt += f"システム: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"ユーザー: {content}\n"
            elif role == "assistant":
                formatted_prompt += f"アシスタント: {content}\n"
        
        formatted_prompt += "アシスタント: "
        return formatted_prompt

def interactive_chat():
    """ユーザー入力による対話"""
    
    chat = LocalModelChat()
    
    print("=" * 60)
    print("🏠 ローカルLLM 対話テスト（OpenAI API不要）")
    print("=" * 60)
    print("💬 何でも質問してください！")
    print("終了するには 'exit' と入力")
    print("リセットするには 'reset' と入力")
    print("-" * 60)
    
    # 初期システムメッセージ
    messages = [
        {
            "role": "system",
            "content": "あなたは親切で知識豊富なアシスタントです。日本語で丁寧に対応してください。"
        }
    ]
    
    while True:
        # ユーザー入力
        user_input = input("\n👤 あなた: ").strip()
        
        # 終了チェック
        if user_input.lower() in ['exit', 'quit', '終了', 'bye']:
            print("\n👋 お疲れさまでした！")
            break
        
        # リセットチェック
        if user_input.lower() in ['reset', 'リセット']:
            messages = [
                {
                    "role": "system",
                    "content": "あなたは親切で知識豊富なアシスタントです。日本語で丁寧に対応してください。"
                }
            ]
            print("🔄 会話履歴をリセットしました")
            continue
        
        # 空入力をスキップ
        if not user_input:
            continue
        
        # 現在の会話にユーザー入力を追加
        current_messages = messages + [{"role": "user", "content": user_input}]
        
        print("\n🤖 ローカルAI: ", end="", flush=True)
        
        # ローカルモデルから応答を取得
        response = chat.chat_with_local_model(current_messages)
        print(response)
        
        # 会話履歴を更新
        messages.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response}
        ])
        
        # 履歴が長くなりすぎたら古いものを削除（メモリ節約）
        if len(messages) > 11:  # システム + 10メッセージまで
            messages = [messages[0]] + messages[-10:]

def show_setup_instructions():
    """セットアップ方法を表示"""
    
    print("""
🚀 Ollama セットアップ方法:

【1. Ollamaインストール】
curl -fsSL https://ollama.com/install.sh | sh

【2. Ollamaサーバー起動】
ollama serve

【3. 軽量モデルをダウンロード】
ollama pull llama3.2:3b        # 約2GB
# または
ollama pull qwen2:1.5b         # 約1.5GB (軽い)
ollama pull phi3:mini          # 約2.3GB

【4. このプログラム実行】
python test_local_interactive.py

💡 ヒント:
- Ollamaは別のターミナルで 'ollama serve' で起動
- モデルは初回ダウンロード後はローカルに保存
- インターネット接続不要で動作（ダウンロード後）
""")

if __name__ == "__main__":
    print("🏠 ローカルLLM 対話テストプログラム")
    print("OpenAI APIキーは不要です")
    
    # まずOllamaの状態をチェック
    chat = LocalModelChat()
    
    if chat.is_ollama_running():
        print("✅ Ollama接続OK - 対話開始")
        interactive_chat()
    else:
        print("⚠️  Ollamaが起動していません")
        show_setup_instructions()
        
        choice = input("\nOllamaを起動済みの場合、対話を開始しますか？ (y/N): ").strip().lower()
        if choice == 'y':
            interactive_chat()