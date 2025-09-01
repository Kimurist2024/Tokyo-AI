#!/usr/bin/env python3
"""
Check available OpenAI models
利用可能なOpenAIモデルを確認
"""

import openai
import os
from dotenv import load_dotenv

load_dotenv()

def check_available_models():
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        models = client.models.list()
        
        # Chat completionに使用できるモデルをフィルタ
        chat_models = []
        for model in models.data:
            model_id = model.id
            if any(prefix in model_id for prefix in ["gpt-", "o1-"]):
                chat_models.append(model_id)
        
        print("🤖 Available OpenAI Chat Models:")
        print("=" * 50)
        
        # 推奨順で並び替え
        recommended_order = [
            "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", 
            "gpt-3.5-turbo", "o1-preview", "o1-mini"
        ]
        
        # 推奨モデルから表示
        for model in recommended_order:
            if model in chat_models:
                print(f"✅ {model}")
                chat_models.remove(model)
        
        # その他のモデル
        if chat_models:
            print("\n📋 Other available models:")
            for model in sorted(chat_models):
                print(f"   {model}")
        
        print(f"\n💡 Total models available: {len(chat_models) + len(recommended_order)}")
        
    except Exception as e:
        print(f"❌ Error checking models: {e}")

if __name__ == "__main__":
    check_available_models()