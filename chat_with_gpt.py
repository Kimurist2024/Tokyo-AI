"""
OpenAI ChatGPT APIを使った対話形式のチャットプログラム
"""

from api_key_manager import APIKeyManager
from openai import OpenAI
import sys

def chat_with_gpt():
    """ChatGPTと対話形式で会話する"""
    
    # APIキーマネージャーの初期化
    manager = APIKeyManager()
    api_key = manager.get_key("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OpenAI APIキーが設定されていません")
        print("   .envファイルにOPENAI_API_KEYを設定してください")
        return
    
    # OpenAIクライアントの初期化
    client = OpenAI(api_key=api_key)
    
    print("=" * 60)
    print("🤖 ChatGPTとの対話を開始します")
    print("=" * 60)
    print("終了するには 'exit', 'quit', または 'bye' と入力してください")
    print("会話をリセットするには 'reset' と入力してください")
    print("-" * 60)
    
    # 会話履歴を保持
    messages = [
        {"role": "system", "content": "あなたは親切で役立つアシスタントです。日本語で自然に会話してください。"}
    ]
    
    while True:
        # ユーザー入力を取得
        user_input = input("\n👤 あなた: ").strip()
        
        # 終了コマンドをチェック
        if user_input.lower() in ['exit', 'quit', 'bye', '終了', 'さようなら']:
            print("\n👋 ChatGPT: さようなら！またお話ししましょう！")
            break
        
        # リセットコマンドをチェック
        if user_input.lower() == 'reset':
            messages = [
                {"role": "system", "content": "あなたは親切で役立つアシスタントです。日本語で自然に会話してください。"}
            ]
            print("\n🔄 会話履歴をリセットしました")
            continue
        
        # 空の入力をスキップ
        if not user_input:
            continue
        
        # ユーザーメッセージを追加
        messages.append({"role": "user", "content": user_input})
        
        try:
            # ChatGPT APIを呼び出し
            print("\n🤖 ChatGPT: ", end="", flush=True)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.8,
                stream=True  # ストリーミングレスポンスを有効化
            )
            
            # レスポンスをストリーミング表示
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print()  # 改行
            
            # アシスタントの応答を会話履歴に追加
            messages.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            print("\n\n⚠️  中断されました")
            continue
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")
            print("もう一度お試しください。")


def main():
    """メイン関数"""
    try:
        chat_with_gpt()
    except KeyboardInterrupt:
        print("\n\n👋 プログラムを終了します")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()