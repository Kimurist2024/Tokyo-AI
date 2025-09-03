"""
GPT-5 nanoのファインチューニング実行スクリプト
"""

import json
import time
from pathlib import Path
from api_key_manager import APIKeyManager
from openai import OpenAI

class GPT5NanoFineTuner:
    def __init__(self):
        manager = APIKeyManager()
        api_key = manager.get_key("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI APIキーが設定されていません")
        
        self.client = OpenAI(api_key=api_key)
        self.training_file_id = None
        self.validation_file_id = None
        self.job_id = None
    
    def upload_dataset(self, train_file: Path, val_file: Path = None):
        """データセットをOpenAIにアップロード"""
        
        print("📤 データセットをアップロード中...")
        
        # トレーニングファイルをアップロード
        with open(train_file, "rb") as f:
            response = self.client.files.create(
                file=f,
                purpose="fine-tune"
            )
            self.training_file_id = response.id
            print(f"✅ トレーニングファイル: {self.training_file_id}")
        
        # 検証ファイルをアップロード（オプション）
        if val_file and val_file.exists():
            with open(val_file, "rb") as f:
                response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
                self.validation_file_id = response.id
                print(f"✅ 検証ファイル: {self.validation_file_id}")
        
        return self.training_file_id, self.validation_file_id
    
    def start_finetuning(self, model="gpt-5-nano", suffix="travel-jp"):
        """ファインチューニングジョブを開始"""
        
        print(f"\n🚀 {model}のファインチューニングを開始...")
        
        try:
            # ファインチューニングジョブを作成
            job_params = {
                "training_file": self.training_file_id,
                "model": model,
                "suffix": suffix
            }
            
            # 検証ファイルがある場合は追加
            if self.validation_file_id:
                job_params["validation_file"] = self.validation_file_id
            
            # ハイパーパラメータの設定（GPT-5 nano用に調整）
            job_params["hyperparameters"] = {
                "n_epochs": 3  # エポック数を少なめに
            }
            
            response = self.client.fine_tuning.jobs.create(**job_params)
            self.job_id = response.id
            
            print(f"✅ ファインチューニングジョブ開始: {self.job_id}")
            print(f"   モデル: {model}")
            print(f"   サフィックス: {suffix}")
            
            return self.job_id
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            
            # エラーメッセージを解析
            error_str = str(e)
            if "not supported" in error_str.lower() or "invalid" in error_str.lower():
                print("\n⚠️  GPT-5 nanoはファインチューニングに対応していない可能性があります")
                print("\n💡 代替案：")
                print("  1. gpt-4o-mini（推奨）でファインチューニング")
                print("  2. gpt-3.5-turboでファインチューニング")
                
                # 代替モデルで再試行するか確認
                return self.fallback_to_alternative_model()
            
            raise e
    
    def fallback_to_alternative_model(self):
        """代替モデルでファインチューニング"""
        
        print("\n🔄 gpt-4o-miniで再試行します...")
        
        try:
            job_params = {
                "training_file": self.training_file_id,
                "model": "gpt-4o-mini-2024-07-18",  # 具体的なバージョンを指定
                "suffix": "travel-jp"
            }
            
            if self.validation_file_id:
                job_params["validation_file"] = self.validation_file_id
            
            response = self.client.fine_tuning.jobs.create(**job_params)
            self.job_id = response.id
            
            print(f"✅ ファインチューニングジョブ開始: {self.job_id}")
            print(f"   モデル: gpt-4o-mini")
            
            return self.job_id
            
        except Exception as e:
            print(f"❌ 代替モデルでもエラー: {e}")
            return None
    
    def monitor_job(self):
        """ファインチューニングジョブの進行状況を監視"""
        
        if not self.job_id:
            print("❌ ジョブIDが設定されていません")
            return
        
        print(f"\n📊 ジョブの進行状況を監視中: {self.job_id}")
        print("   (Ctrl+Cで監視を中止)")
        
        try:
            while True:
                job = self.client.fine_tuning.jobs.retrieve(self.job_id)
                
                print(f"\r状態: {job.status}", end="")
                
                if job.status == "succeeded":
                    print(f"\n✅ ファインチューニング完了！")
                    print(f"   モデルID: {job.fine_tuned_model}")
                    return job.fine_tuned_model
                
                elif job.status == "failed":
                    print(f"\n❌ ファインチューニング失敗")
                    if job.error:
                        print(f"   エラー: {job.error}")
                    return None
                
                elif job.status == "cancelled":
                    print(f"\n⚠️  ファインチューニングがキャンセルされました")
                    return None
                
                time.sleep(10)  # 10秒ごとにチェック
                
        except KeyboardInterrupt:
            print("\n\n⚠️  監視を中止しました")
            print(f"   ジョブは継続中です: {self.job_id}")
            print("   後で確認するには: python check_finetune_status.py")
            return None
    
    def test_finetuned_model(self, model_id):
        """ファインチューニング済みモデルをテスト"""
        
        print(f"\n🧪 ファインチューニング済みモデルをテスト: {model_id}")
        
        test_prompts = [
            "東京から大阪への移動方法を教えてください。",
            "北海道旅行のおすすめ時期は？",
            "JRパスの使い方を教えてください。"
        ]
        
        for prompt in test_prompts:
            print(f"\n👤 質問: {prompt}")
            
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=150
                )
                
                print(f"🤖 回答: {response.choices[0].message.content}")
                
            except Exception as e:
                print(f"❌ エラー: {e}")


def main():
    """メイン実行関数"""
    
    print("=" * 60)
    print("🎯 GPT-5 nanoファインチューニング")
    print("=" * 60)
    
    # データセットの準備
    from prepare_travel_dataset import create_travel_dataset
    train_file, val_file = create_travel_dataset()
    
    # ファインチューナーの初期化
    tuner = GPT5NanoFineTuner()
    
    # データセットのアップロード
    tuner.upload_dataset(train_file, val_file)
    
    # ファインチューニング開始
    job_id = tuner.start_finetuning(model="gpt-5-nano")
    
    if job_id:
        # ジョブの監視
        model_id = tuner.monitor_job()
        
        if model_id:
            # モデルのテスト
            tuner.test_finetuned_model(model_id)
            
            print("\n" + "=" * 60)
            print("✅ ファインチューニング完了！")
            print(f"   モデルID: {model_id}")
            print("\n使用方法:")
            print(f'   model="{model_id}"')
            print("=" * 60)
        else:
            print("\n⚠️  ファインチューニングが完了していません")
            print(f"   ジョブID: {job_id}")
            print("   後で状態を確認してください")
    else:
        print("\n❌ ファインチューニングを開始できませんでした")


if __name__ == "__main__":
    main()