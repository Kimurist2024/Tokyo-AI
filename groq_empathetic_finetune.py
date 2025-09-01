"""
Groqを使用した共感的応答の学習・改善システム
ユーザーとの対話を通じて応答品質を向上させる
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class EmpatheticResponseTrainer:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.training_data = []
        self.load_training_data()
    
    def load_training_data(self, filename="empathetic_training_data.json"):
        """既存の訓練データを読み込み"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)
            print(f"既存の訓練データを読み込みました: {len(self.training_data)}件")
        except FileNotFoundError:
            print("新しい訓練データファイルを作成します")
            self.training_data = []
    
    def save_training_data(self, filename="empathetic_training_data.json"):
        """訓練データを保存"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        print(f"訓練データを保存しました: {len(self.training_data)}件")
    
    def generate_improved_response(self, user_message, previous_responses=None):
        """より共感的な応答を生成"""
        
        # プロンプトエンジニアリングで応答品質を向上
        enhanced_prompt = f"""あなたは最高レベルの共感力を持つカウンセラーです。以下のメッセージに対して、心の底から寄り添う応答をしてください。

ユーザーのメッセージ: "{user_message}"

応答の際は以下の要素を含めてください：
1. 感情の承認: ユーザーの感情を明確に認識し、受け入れる
2. 共感の表現: "辛いですね"、"大変でしたね" などの共感語を使用
3. 具体的な理解: 状況の具体的な側面に言及
4. 支持的なメッセージ: ユーザーを支える言葉
5. 希望や前進のヒント: 可能であれば建設的な提案

温かく、判断せず、ユーザーが理解され大切にされていると感じられる応答をしてください。"""
        
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": enhanced_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.8,
                max_tokens=1024,
                top_p=0.95,
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            return f"応答生成エラー: {str(e)}"
    
    def collect_feedback(self, user_message, response):
        """ユーザーからのフィードバックを収集"""
        print(f"\nAI応答: {response}")
        print("\n" + "-"*50)
        print("この応答はどうでしたか？")
        print("1: とても良い (Very Good)")
        print("2: 良い (Good)")  
        print("3: 普通 (Average)")
        print("4: 改善が必要 (Needs Improvement)")
        print("5: 良くない (Poor)")
        
        while True:
            try:
                rating = int(input("評価を入力 (1-5): "))
                if 1 <= rating <= 5:
                    break
                else:
                    print("1-5の数字を入力してください")
            except ValueError:
                print("数字を入力してください")
        
        feedback = input("改善提案があれば教えてください (optional): ").strip()
        
        # 訓練データに追加
        training_sample = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "ai_response": response,
            "rating": rating,
            "feedback": feedback,
            "category": self.categorize_emotion(user_message)
        }
        
        self.training_data.append(training_sample)
        return rating, feedback
    
    def categorize_emotion(self, message):
        """メッセージの感情カテゴリを分析"""
        emotion_keywords = {
            "sadness": ["悲しい", "辛い", "泣きたい", "憂鬱", "落ち込む"],
            "anxiety": ["不安", "心配", "怖い", "緊張", "ドキドキ"],
            "anger": ["怒り", "イライラ", "腹立たしい", "ムカつく"],
            "loneliness": ["寂しい", "孤独", "一人", "独り"],
            "stress": ["疲れた", "ストレス", "忙しい", "大変", "きつい"],
            "disappointment": ["失望", "がっかり", "期待外れ", "ショック"]
        }
        
        message_lower = message.lower()
        for category, keywords in emotion_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return category
        
        return "general"
    
    def analyze_performance(self):
        """応答性能の分析"""
        if not self.training_data:
            print("分析するデータがありません")
            return
        
        # 評価の統計
        ratings = [item["rating"] for item in self.training_data]
        avg_rating = sum(ratings) / len(ratings)
        
        # カテゴリ別の性能
        category_stats = {}
        for item in self.training_data:
            cat = item["category"]
            if cat not in category_stats:
                category_stats[cat] = []
            category_stats[cat].append(item["rating"])
        
        print(f"\n{'='*50}")
        print("応答性能分析")
        print(f"{'='*50}")
        print(f"総サンプル数: {len(self.training_data)}")
        print(f"平均評価: {avg_rating:.2f}/5.0")
        
        print(f"\nカテゴリ別性能:")
        for category, ratings in category_stats.items():
            avg_cat_rating = sum(ratings) / len(ratings)
            print(f"  {category}: {avg_cat_rating:.2f}/5.0 ({len(ratings)}件)")
        
        # 改善が必要な応答
        poor_responses = [item for item in self.training_data if item["rating"] <= 2]
        if poor_responses:
            print(f"\n改善が必要な応答: {len(poor_responses)}件")
            for item in poor_responses[-3:]:  # 最新3件表示
                print(f"  メッセージ: {item['user_message'][:50]}...")
                print(f"  評価: {item['rating']}/5")
                if item["feedback"]:
                    print(f"  フィードバック: {item['feedback']}")
                print()
    
    def interactive_training(self):
        """対話的な訓練セッション"""
        print("\n" + "="*60)
        print("共感的AI応答の改善訓練システム")
        print("ユーザーからのフィードバックで応答品質を向上させます")
        print("'quit'で終了、'analyze'で性能分析")
        print("="*60 + "\n")
        
        while True:
            user_input = input("テストしたいメッセージを入力: ").strip()
            
            if user_input.lower() == 'quit':
                self.save_training_data()
                print("訓練セッションを終了します")
                break
            
            if user_input.lower() == 'analyze':
                self.analyze_performance()
                continue
            
            if not user_input:
                continue
            
            # 改善された応答を生成
            response = self.generate_improved_response(user_input)
            
            # フィードバックを収集
            rating, feedback = self.collect_feedback(user_input, response)
            
            # 即座に保存
            self.save_training_data()
            
            print(f"フィードバックありがとうございます！(評価: {rating}/5)")
            print()

def main():
    trainer = EmpatheticResponseTrainer()
    trainer.interactive_training()

if __name__ == "__main__":
    main()