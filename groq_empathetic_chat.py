"""
Groqを使用した共感的なAIチャットボット
人に寄り添う応答を提供するように調整
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

# .envファイルから環境変数を読み込み
load_dotenv()

class EmpatheticGroqChat:
    def __init__(self):
        """共感的なGroqチャットボットの初期化"""
        self.client = self.setup_groq_client()
        self.conversation_history = []
        
        # 共感的な応答のためのシステムプロンプト
        self.system_prompt = """あなたは非常に共感的で思いやりのあるカウンセラーのようなAIアシスタントです。

以下のガイドラインに従って応答してください:

1. **共感を示す**: ユーザーの感情を理解し、その気持ちを受け入れる
2. **支持的な態度**: 批判せず、判断せず、ユーザーを支える
3. **励ましと希望**: 困難な状況でも前向きな要素を見つけて励ます
4. **具体的なサポート**: 可能であれば実用的なアドバイスや提案を提供
5. **温かい口調**: 優しく、理解のある言葉を使用する

ユーザーが悲しみ、不安、怒り、失望などの感情を表現した時は、特に注意深く共感的に応答してください。"""

    def setup_groq_client(self):
        """Groq APIクライアントのセットアップ"""
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            print("環境変数 GROQ_API_KEY を設定してください")
            print("1. https://console.groq.com でアカウント作成")
            print("2. APIキーを取得")
            print("3. .envファイルに GROQ_API_KEY=your-api-key を追加")
            return None
        
        return Groq(api_key=api_key)

    def get_empathetic_response(self, user_message):
        """共感的な応答を生成"""
        if not self.client:
            return "APIクライアントが初期化されていません"
        
        try:
            # 会話履歴を含めたメッセージ作成
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # 最近の会話履歴を追加（最大5回の対話）
            recent_history = self.conversation_history[-10:]  # 最新10メッセージ
            messages.extend(recent_history)
            
            # 現在のユーザーメッセージを追加
            messages.append({"role": "user", "content": user_message})
            
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.7,  # 創造性と一貫性のバランス
                max_tokens=1024,
                top_p=0.9,
            )
            
            response = completion.choices[0].message.content
            
            # 会話履歴を更新
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            return f"申し訳ありません、エラーが発生しました: {str(e)}"

    def save_conversation(self, filename="conversation_history.json"):
        """会話履歴を保存"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"会話履歴を {filename} に保存しました")
        except Exception as e:
            print(f"保存エラー: {str(e)}")

    def load_conversation(self, filename="conversation_history.json"):
        """会話履歴を読み込み"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"{filename} から会話履歴を読み込みました")
        except FileNotFoundError:
            print("会話履歴ファイルが見つかりません。新しい会話を開始します。")
        except Exception as e:
            print(f"読み込みエラー: {str(e)}")

    def chat_loop(self):
        """メインのチャットループ"""
        if not self.client:
            return
        
        print("\n" + "="*60)
        print("共感的なAIチャットボット - Powered by Groq")
        print("あなたの気持ちに寄り添って対話します")
        print("'quit', 'exit', 'q' で終了")
        print("'save' で会話を保存、'load' で会話を読み込み")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("あなた: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nお疲れさまでした。また何かあればいつでもお話しください。")
                    break
                
                if user_input.lower() == 'save':
                    self.save_conversation()
                    continue
                
                if user_input.lower() == 'load':
                    self.load_conversation()
                    continue
                
                if not user_input:
                    continue
                
                print("AI: ", end="", flush=True)
                response = self.get_empathetic_response(user_input)
                print(response + "\n")
                
            except KeyboardInterrupt:
                print("\n\nチャットを終了します。お疲れさまでした。")
                break
            except Exception as e:
                print(f"\nエラーが発生しました: {str(e)}\n")

def test_empathetic_responses():
    """共感的応答のテスト"""
    chat = EmpatheticGroqChat()
    
    if not chat.client:
        print("APIクライアントの初期化に失敗しました")
        return
    
    test_messages = [
        "今日はとても疲れてしまいました...",
        "友達と喧嘩してしまって悲しいです",
        "仕事が上手くいかなくて落ち込んでいます",
        "最近寝れなくて辛いです",
        "試験に落ちてしまいました"
    ]
    
    print("="*50)
    print("共感的応答テスト")
    print("="*50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n【テスト{i}】")
        print(f"ユーザー: {message}")
        response = chat.get_empathetic_response(message)
        print(f"AI: {response}")
        print("-" * 40)

def main():
    """メイン関数"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_empathetic_responses()
    else:
        chat = EmpatheticGroqChat()
        chat.chat_loop()

if __name__ == "__main__":
    main()