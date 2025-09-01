"""
Ollama を使用した完全無料のローカルAIチャット
自前のマシンで動作、API料金不要
"""

import requests
import json

class OllamaChat:
    def __init__(self, model="llama3", base_url="http://localhost:11434"):
        """
        Ollamaのセットアップ
        インストール: curl -fsSL https://ollama.com/install.sh | sh
        モデル取得: ollama pull llama3
        """
        self.model = model
        self.base_url = base_url
        self.conversation_history = []
    
    def chat(self, user_message):
        """
        Ollamaでチャット応答を生成
        """
        self.conversation_history.append({"role": "user", "content": user_message})
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": self.conversation_history,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['message']['content']
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                return ai_response
            else:
                return f"エラー: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Ollamaサーバーに接続できません。'ollama serve'を実行してください。"
        except Exception as e:
            return f"エラー: {str(e)}"

def main():
    print("=== Ollama ローカルAIチャット ===")
    print("セットアップ手順:")
    print("1. Ollamaインストール: curl -fsSL https://ollama.com/install.sh | sh")
    print("2. モデル取得: ollama pull llama3")
    print("3. サーバー起動: ollama serve")
    print("=====================================\n")
    
    chat = OllamaChat(model="llama3")
    
    print("チャット開始（'quit'で終了）\n")
    
    while True:
        user_input = input("あなた: ")
        if user_input.lower() == 'quit':
            break
        
        response = chat.chat(user_input)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    main()