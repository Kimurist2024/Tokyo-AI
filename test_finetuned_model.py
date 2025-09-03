"""
ファインチューニング済みモデルのテスト
"""

from api_key_manager import APIKeyManager
from openai import OpenAI

def test_finetuned_model():
    """ファインチューニング済みモデルをテスト"""
    
    # ファインチューニング済みモデルID
    model_id = "ft:gpt-4o-mini-2024-07-18:kimurist:travel-jp-gpu:CBaBln2U"
    
    manager = APIKeyManager()
    client = OpenAI(api_key=manager.get_key("OPENAI_API_KEY"))
    
    print("🧪 ファインチューニング済みモデルをテスト")
    print(f"   モデルID: {model_id}")
    
    # テストプロンプト（旅行関連）
    test_prompts = [
        "東京から大阪への移動方法を教えてください。",
        "北海道旅行のおすすめ時期はいつですか？",
        "JRパスの使い方を教えてください。",
        "富士山に登るのに必要な装備は？",
        "京都の紅葉の見頃を教えてください。",
        "成田空港から都心への移動方法は？"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n【テスト {i}/{len(test_prompts)}】")
        print(f"👤 質問: {prompt}")
        
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system", 
                        "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。日本語で丁寧に対応してください。"
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            print(f"🤖 回答: {response.choices[0].message.content}")
            print(f"   使用トークン: {response.usage.total_tokens}")
            
        except Exception as e:
            print(f"❌ エラー: {e}")

def chat_with_finetuned_model():
    """ファインチューニング済みモデルとの対話"""
    
    model_id = "ft:gpt-4o-mini-2024-07-18:kimurist:travel-jp-gpu:CBaBln2U"
    
    manager = APIKeyManager()
    client = OpenAI(api_key=manager.get_key("OPENAI_API_KEY"))
    
    print("\n" + "=" * 60)
    print("💬 ファインチューニング済みモデルとの対話")
    print("=" * 60)
    print("終了するには 'exit' と入力してください")
    print("-" * 60)
    
    # 会話履歴
    messages = [
        {
            "role": "system",
            "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。日本語で丁寧に対応してください。"
        }
    ]
    
    while True:
        user_input = input("\n👤 あなた: ").strip()
        
        if user_input.lower() in ['exit', 'quit', '終了']:
            print("\n👋 ありがとうございました！")
            break
        
        if not user_input:
            continue
        
        # ユーザーメッセージを追加
        messages.append({"role": "user", "content": user_input})
        
        try:
            print("\n🤖 旅行エージェント: ", end="", flush=True)
            
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=300,
                temperature=0.7,
                stream=True
            )
            
            # ストリーミング表示
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print()  # 改行
            
            # 応答を履歴に追加
            messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            print(f"\n❌ エラー: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 ファインチューニング済みモデル テスト & 対話")
    print("=" * 60)
    
    # まずテストを実行
    test_finetuned_model()
    
    # 対話モード開始
    chat_with_finetuned_model()