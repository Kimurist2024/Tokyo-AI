#!/usr/bin/env python3
"""
Chatbot Evaluation Script with Metrics
チャットボットの評価スクリプト
"""

from empathetic_chat_advanced import AdvancedEmpatheticChatBot
import wandb
import os
from dotenv import load_dotenv
import time
import json
from typing import List, Dict
import numpy as np

# 環境変数の読み込み
load_dotenv()

# 評価用のテストケース
EVALUATION_TEST_CASES = [
    {
        "input": "今日はとても疲れました。仕事が大変で...",
        "expected_keywords": ["お疲れ", "大変", "休", "理解"],
        "category": "fatigue"
    },
    {
        "input": "友達と喧嘩してしまって悲しいです",
        "expected_keywords": ["悲し", "理解", "気持ち", "友"],
        "category": "relationship"
    },
    {
        "input": "仕事が上手くいかなくて落ち込んでいます",
        "expected_keywords": ["落ち込", "仕事", "大丈夫", "支援"],
        "category": "work"
    },
    {
        "input": "最近寝れなくて辛いです",
        "expected_keywords": ["辛", "睡眠", "休", "心配"],
        "category": "health"
    },
    {
        "input": "試験に落ちてしまいました",
        "expected_keywords": ["試験", "次", "頑張", "大丈夫"],
        "category": "failure"
    },
    {
        "input": "I'm feeling very anxious about my future",
        "expected_keywords": ["understand", "anxious", "future", "support"],
        "category": "anxiety"
    },
    {
        "input": "I feel so lonely these days",
        "expected_keywords": ["lonely", "understand", "care", "here"],
        "category": "loneliness"
    }
]

class ChatbotEvaluator:
    def __init__(self, model_name: str = "mistral-7b", use_wandb: bool = True):
        """
        評価クラスの初期化
        """
        self.chatbot = AdvancedEmpatheticChatBot(
            model_name=model_name,
            use_wandb=use_wandb,
            project_name="chatbot-evaluation"
        )
        self.results = []
        self.use_wandb = use_wandb
    
    def evaluate_empathy_score(self, response: str, expected_keywords: List[str]) -> float:
        """
        共感度スコアの計算
        """
        response_lower = response.lower()
        matched_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        score = matched_keywords / len(expected_keywords) if expected_keywords else 0
        return score
    
    def evaluate_response_quality(self, response: str) -> Dict:
        """
        応答品質の評価
        """
        quality_metrics = {
            "length": len(response),
            "word_count": len(response.split()),
            "sentence_count": response.count('。') + response.count('.') + response.count('!') + response.count('?'),
            "has_greeting": any(greet in response.lower() for greet in ["こんにちは", "hello", "hi"]),
            "has_question": '?' in response or '？' in response,
            "has_encouragement": any(enc in response.lower() for enc in ["頑張", "大丈夫", "できる", "you can", "it's okay", "will be"])
        }
        
        # 品質スコア（0-1）
        quality_score = 0
        if 50 <= quality_metrics["length"] <= 500:
            quality_score += 0.25
        if quality_metrics["word_count"] >= 10:
            quality_score += 0.25
        if quality_metrics["has_question"]:
            quality_score += 0.25
        if quality_metrics["has_encouragement"]:
            quality_score += 0.25
        
        quality_metrics["quality_score"] = quality_score
        
        return quality_metrics
    
    def run_evaluation(self, test_cases: List[Dict] = None) -> Dict:
        """
        評価の実行
        """
        if test_cases is None:
            test_cases = EVALUATION_TEST_CASES
        
        print("\n" + "="*60)
        print("🔬 Starting Chatbot Evaluation")
        print(f"Model: {self.chatbot.model_name}")
        print(f"Test cases: {len(test_cases)}")
        print("="*60 + "\n")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Testing: {test_case['category']}")
            print(f"Input: {test_case['input']}")
            
            # 応答生成
            response, metrics = self.chatbot.generate_response(
                test_case['input'],
                max_length=300,
                temperature=0.7
            )
            
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
            
            # 評価
            empathy_score = self.evaluate_empathy_score(response, test_case.get('expected_keywords', []))
            quality_metrics = self.evaluate_response_quality(response)
            
            # 結果の保存
            result = {
                "test_case": i,
                "category": test_case['category'],
                "input": test_case['input'],
                "response": response,
                "empathy_score": empathy_score,
                "response_time": metrics['response_time'],
                **quality_metrics,
                **metrics
            }
            
            self.results.append(result)
            
            print(f"✓ Empathy Score: {empathy_score:.2f}")
            print(f"✓ Quality Score: {quality_metrics['quality_score']:.2f}")
            print(f"✓ Response Time: {metrics['response_time']:.2f}s")
            print("-" * 40)
            
            # WandBにログ
            if self.use_wandb:
                wandb.log({
                    f"eval/{test_case['category']}/empathy_score": empathy_score,
                    f"eval/{test_case['category']}/quality_score": quality_metrics['quality_score'],
                    f"eval/{test_case['category']}/response_time": metrics['response_time']
                })
        
        # 総合評価
        overall_metrics = self.calculate_overall_metrics()
        
        return overall_metrics
    
    def calculate_overall_metrics(self) -> Dict:
        """
        全体的なメトリクスの計算
        """
        if not self.results:
            return {}
        
        overall = {
            "total_tests": len(self.results),
            "avg_empathy_score": np.mean([r['empathy_score'] for r in self.results]),
            "avg_quality_score": np.mean([r['quality_score'] for r in self.results]),
            "avg_response_time": np.mean([r['response_time'] for r in self.results]),
            "avg_response_length": np.mean([r['length'] for r in self.results]),
            "avg_word_count": np.mean([r['word_count'] for r in self.results]),
            "std_response_time": np.std([r['response_time'] for r in self.results]),
            "min_response_time": min([r['response_time'] for r in self.results]),
            "max_response_time": max([r['response_time'] for r in self.results])
        }
        
        # カテゴリ別の集計
        categories = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        category_metrics = {}
        for cat, results in categories.items():
            category_metrics[cat] = {
                "avg_empathy": np.mean([r['empathy_score'] for r in results]),
                "avg_quality": np.mean([r['quality_score'] for r in results]),
                "avg_time": np.mean([r['response_time'] for r in results])
            }
        
        overall["category_metrics"] = category_metrics
        
        # WandBにサマリーをログ
        if self.use_wandb:
            wandb.log({"overall": overall})
        
        return overall
    
    def save_results(self, filename: str = "evaluation_results.json"):
        """
        評価結果の保存
        """
        output = {
            "model": self.chatbot.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": self.results,
            "overall_metrics": self.calculate_overall_metrics()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 Results saved to {filename}")
    
    def print_summary(self):
        """
        評価サマリーの表示
        """
        metrics = self.calculate_overall_metrics()
        
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\n📈 Overall Metrics:")
        print(f"  • Total Tests: {metrics['total_tests']}")
        print(f"  • Avg Empathy Score: {metrics['avg_empathy_score']:.3f}")
        print(f"  • Avg Quality Score: {metrics['avg_quality_score']:.3f}")
        print(f"  • Avg Response Time: {metrics['avg_response_time']:.2f}s ± {metrics['std_response_time']:.2f}s")
        print(f"  • Avg Response Length: {metrics['avg_response_length']:.0f} chars")
        
        print(f"\n📂 Category Performance:")
        for cat, cat_metrics in metrics['category_metrics'].items():
            print(f"  {cat}:")
            print(f"    - Empathy: {cat_metrics['avg_empathy']:.3f}")
            print(f"    - Quality: {cat_metrics['avg_quality']:.3f}")
            print(f"    - Time: {cat_metrics['avg_time']:.2f}s")
        
        # パフォーマンス評価
        print(f"\n🎯 Performance Rating:")
        overall_score = (metrics['avg_empathy_score'] + metrics['avg_quality_score']) / 2
        if overall_score >= 0.8:
            rating = "Excellent ⭐⭐⭐⭐⭐"
        elif overall_score >= 0.6:
            rating = "Good ⭐⭐⭐⭐"
        elif overall_score >= 0.4:
            rating = "Fair ⭐⭐⭐"
        else:
            rating = "Needs Improvement ⭐⭐"
        
        print(f"  Overall Score: {overall_score:.3f} - {rating}")
        print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Empathetic ChatBot")
    parser.add_argument("--model", default="mistral-7b",
                       help="Model to evaluate")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable WandB logging")
    parser.add_argument("--save", default="evaluation_results.json",
                       help="Output filename for results")
    
    args = parser.parse_args()
    
    # 評価の実行
    evaluator = ChatbotEvaluator(
        model_name=args.model,
        use_wandb=not args.no_wandb
    )
    
    # 評価実行
    overall_metrics = evaluator.run_evaluation()
    
    # サマリー表示
    evaluator.print_summary()
    
    # 結果保存
    evaluator.save_results(args.save)
    
    # WandB終了
    if evaluator.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()