"""
GPT-5 nanoモデルのテスト
"""

from api_key_manager import APIKeyManager
from openai import OpenAI

def test_gpt5_nano():
    """GPT-5 nanoモデルをテスト"""
    
    manager = APIKeyManager()
    api_key = manager.get_key("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OpenAI APIキーが設定されていません")
        return
    
    client = OpenAI(api_key=api_key)
    
    print("🚀 GPT-5 nanoモデルをテスト中...\n")
    
    try:
        # GPT-5 nanoでチャット
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "あなたは親切な旅行アドバイザーです。"},
                {"role": "user", "content": "東京の観光スポットを3つ教えてください。"}
            ],
            max_completion_tokens=200
        )
        
        print("✅ GPT-5 nanoの応答:")
        print(response.choices[0].message.content)
        
        # モデル情報を表示
        print(f"\n📊 使用モデル: {response.model}")
        print(f"トークン使用量: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        
        # 代替モデルを提案
        print("\n💡 代替案: GPT-4o-miniを使用してファインチューニングを行います")
        return False


if __name__ == "__main__":
    success = test_gpt5_nano()
    
    if not success:
        print("\n次のモデルでファインチューニングが可能です:")
        print("  • gpt-4o-mini (推奨)")
        print("  • gpt-3.5-turbo")
        print("  • gpt-4o")