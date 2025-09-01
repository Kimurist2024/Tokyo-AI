"""
Groq API を使用した高速AIチャットの実装例
無料で始められ、非常に高速なレスポンスが特徴
"""

import os
from dotenv import load_dotenv
from groq import Groq

# .envファイルから環境変数を読み込み
load_dotenv()

def setup_groq_chat():
    """
    Groq APIのセットアップ
    APIキーは https://console.groq.com で無料取得可能
    """
    
    # 環境変数からAPIキーを取得（.envファイルに保存推奨）
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("環境変数 GROQ_API_KEY を設定してください")
        print("1. https://console.groq.com でアカウント作成")
        print("2. APIキーを取得")
        print("3. export GROQ_API_KEY='your-api-key' を実行")
        return None
    
    return Groq(api_key=api_key)

def chat_with_groq(client, user_message):
    """
    Groqでチャット応答を生成
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # または "mixtral-8x7b-32768"
            messages=[
                {"role": "system", "content": "あなたは親切なアシスタントです。"},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"エラー: {str(e)}"

def main():
    # Groqクライアントの初期化
    client = setup_groq_chat()
    if not client:
        return
    
    print("Groq AIチャットを開始します（'quit'で終了）\n")
    
    while True:
        user_input = input("あなた: ")
        if user_input.lower() == 'quit':
            break
        
        response = chat_with_groq(client, user_input)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    main()