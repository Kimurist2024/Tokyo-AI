#!/usr/bin/env python3
"""
Groq API キーの確認プログラム
"""

import os
from dotenv import load_dotenv
from groq import Groq

# .envファイルから環境変数を読み込み
load_dotenv()

def test_groq_api():
    """Groq APIキーの確認とテスト"""
    
    # APIキーの取得
    api_key = os.getenv("GROQ_API_KEY")
    
    print("=== Groq API キー確認 ===")
    
    if not api_key:
        print("❌ APIキーが設定されていません")
        print("   .envファイルにGROQ_API_KEYを設定してください")
        return False
    
    if api_key.startswith("gsk_"):
        print(f"✅ APIキー形式: OK (gsk_で始まっています)")
        print(f"   キーの先頭: {api_key[:20]}...")
    else:
        print("⚠️  APIキー形式が正しくない可能性があります")
        return False
    
    # API接続テスト
    print("\n=== API接続テスト ===")
    try:
        client = Groq(api_key=api_key)
        
        # シンプルなテストメッセージを送信
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # 最新のモデルを使用
            messages=[
                {"role": "user", "content": "こんにちは。これはテストです。"}
            ],
            max_tokens=50,
            temperature=0
        )
        
        response = completion.choices[0].message.content
        print("✅ API接続: 成功")
        print(f"   応答: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ API接続エラー: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_groq_api()
    
    if success:
        print("\n✅ Groq APIの設定が正常に完了しています！")
        print("   groq_example.py を実行してチャットを開始できます。")
    else:
        print("\n❌ 設定を確認してください")