"""
Google Gemini API を使用したAIチャット
無料枠が豊富（60リクエスト/分）
"""

import os
import google.generativeai as genai

def setup_gemini():
    """
    Gemini APIのセットアップ
    APIキーは https://makersuite.google.com/app/apikey で無料取得
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("環境変数 GEMINI_API_KEY を設定してください")
        print("1. https://makersuite.google.com/app/apikey でAPIキー取得")
        print("2. export GEMINI_API_KEY='your-api-key' を実行")
        return None
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')

def chat_with_gemini(model, conversation_history, user_message):
    """
    Geminiでチャット応答を生成
    """
    try:
        # 会話履歴を構築
        prompt = ""
        for msg in conversation_history:
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += f"user: {user_message}\nassistant: "
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"エラー: {str(e)}"

def main():
    # Geminiモデルの初期化
    model = setup_gemini()
    if not model:
        return
    
    print("Google Gemini AIチャット（無料枠: 60リクエスト/分）")
    print("チャット開始（'quit'で終了）\n")
    
    conversation_history = []
    
    while True:
        user_input = input("あなた: ")
        if user_input.lower() == 'quit':
            break
        
        response = chat_with_gemini(model, conversation_history, user_input)
        print(f"AI: {response}\n")
        
        # 会話履歴に追加
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        
        # 履歴が長くなりすぎたら古いものを削除
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

if __name__ == "__main__":
    main()