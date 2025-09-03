"""
OpenAIで利用可能なモデルを確認
"""

from api_key_manager import APIKeyManager
from openai import OpenAI

def check_available_models():
    """利用可能なモデルをリストアップ"""
    
    manager = APIKeyManager()
    api_key = manager.get_key("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OpenAI APIキーが設定されていません")
        return
    
    client = OpenAI(api_key=api_key)
    
    print("🔍 OpenAIで利用可能なモデルを確認中...\n")
    
    # モデル一覧を取得
    models = client.models.list()
    
    # ファインチューニング可能なモデル
    finetune_models = []
    chat_models = []
    
    for model in models:
        model_id = model.id
        
        # GPT関連のモデルのみ表示
        if "gpt" in model_id.lower():
            if "ft:" in model_id:
                continue  # ファインチューニング済みモデルはスキップ
            
            chat_models.append(model_id)
            
            # ファインチューニング可能なモデル
            if any(base in model_id for base in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]):
                finetune_models.append(model_id)
    
    print("📊 チャット用モデル:")
    for model in sorted(set(chat_models)):
        print(f"  • {model}")
    
    print("\n✨ ファインチューニング可能なモデル:")
    for model in sorted(set(finetune_models)):
        if "gpt-4o-mini" in model:
            print(f"  • {model} ← 推奨（最新・高速・安価）")
        elif "gpt-3.5-turbo" in model:
            print(f"  • {model} ← 安定版")
        else:
            print(f"  • {model}")
    
    print("\n💡 メモ:")
    print("  • GPT-4o-mini: 最新の小型高速モデル（GPT-3.5-turboの後継）")
    print("  • GPT-4o: 最新の高性能モデル")
    print("  • GPT-5/GPT-5 nano: まだ公開されていません")
    
    return finetune_models


if __name__ == "__main__":
    check_available_models()