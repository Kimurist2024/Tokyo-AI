"""
JMultiWOZデータでファインチューニング実行
"""

import time
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
from api_key_manager import APIKeyManager
from openai import OpenAI

class JMultiWOZFineTuner:
    def __init__(self):
        manager = APIKeyManager()
        api_key = manager.get_key("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI APIキーが設定されていません")
        
        self.client = OpenAI(api_key=api_key)
        self.training_file_id = None
        self.validation_file_id = None
        self.job_id = None
    
    def upload_jmultiwoz_data(self, train_file: Path, val_file: Path):
        """JMultiWOZデータをアップロード"""
        
        print("📤 JMultiWOZデータをアップロード中...")
        
        # ファイルサイズを取得
        train_size = train_file.stat().st_size
        val_size = val_file.stat().st_size
        
        # トレーニングファイルをアップロード
        with tqdm(total=train_size, unit='B', unit_scale=True, desc="JMultiWOZ Train") as pbar:
            with open(train_file, "rb") as f:
                response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
                self.training_file_id = response.id
                pbar.update(train_size)
        
        print(f"✅ トレーニングファイル: {self.training_file_id}")
        
        # バリデーションファイルをアップロード
        with tqdm(total=val_size, unit='B', unit_scale=True, desc="JMultiWOZ Validation") as pbar:
            with open(val_file, "rb") as f:
                response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
                self.validation_file_id = response.id
                pbar.update(val_size)
        
        print(f"✅ バリデーションファイル: {self.validation_file_id}")
        
        return self.training_file_id, self.validation_file_id
    
    def start_jmultiwoz_finetuning(self, model="gpt-4o-mini-2024-07-18", suffix="jmultiwoz-jp"):
        """JMultiWOZ用ファインチューニング開始"""
        
        print(f"\n🚀 JMultiWOZ ファインチューニング開始...")
        print(f"   モデル: {model}")
        print(f"   データ: 日本語タスク指向対話（700件トレーニング、300件バリデーション）")
        
        try:
            # JMultiWOZ用最適化パラメータ
            job_params = {
                "training_file": self.training_file_id,
                "validation_file": self.validation_file_id,
                "model": model,
                "suffix": suffix,
                "hyperparameters": {
                    "n_epochs": 3,  # 対話データは3エポックが適切
                    "batch_size": 1,  # 長い対話があるためバッチサイズは小さめ
                    "learning_rate_multiplier": 0.1  # 安定した学習のため
                }
            }
            
            response = self.client.fine_tuning.jobs.create(**job_params)
            self.job_id = response.id
            
            print(f"✅ ファインチューニングジョブ開始: {self.job_id}")
            print(f"   サフィックス: {suffix}")
            print(f"   最適化: エポック=3, バッチサイズ=1")
            
            return self.job_id
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            return None
    
    def monitor_jmultiwoz_training(self):
        """JMultiWOZファインチューニングを監視"""
        
        if not self.job_id:
            print("❌ ジョブIDが設定されていません")
            return
        
        print(f"\n📊 JMultiWOZ ファインチューニング監視: {self.job_id}")
        
        start_time = time.time()
        last_status = None
        status_stages = {
            "validating_files": 10,
            "queued": 20,
            "running": 90,
            "succeeded": 100
        }
        
        with tqdm(total=100, desc="JMultiWOZ学習", unit="%", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]') as pbar:
            
            try:
                while True:
                    job = self.client.fine_tuning.jobs.retrieve(self.job_id)
                    
                    if job.status != last_status:
                        last_status = job.status
                        
                        if job.status in status_stages:
                            target_progress = status_stages[job.status]
                            current_progress = pbar.n
                            if target_progress > current_progress:
                                pbar.update(target_progress - current_progress)
                        
                        elapsed = time.time() - start_time
                        elapsed_str = str(timedelta(seconds=int(elapsed)))
                        
                        if job.status == "validating_files":
                            tqdm.write(f"📝 JMultiWOZファイル検証中... (経過: {elapsed_str})")
                        elif job.status == "queued":
                            tqdm.write(f"⏳ キューで待機中... (経過: {elapsed_str})")
                        elif job.status == "running":
                            tqdm.write(f"🎯 JMultiWOZ学習実行中... (経過: {elapsed_str})")
                            # JMultiWOZは対話データなので時間がかかる
                            estimated_total = 1800  # 30分推定
                            remaining = max(0, estimated_total - elapsed)
                            remaining_str = str(timedelta(seconds=int(remaining)))
                            tqdm.write(f"   推定残り時間: {remaining_str}")
                    
                    if job.status == "succeeded":
                        pbar.update(100 - pbar.n)
                        elapsed = time.time() - start_time
                        elapsed_str = str(timedelta(seconds=int(elapsed)))
                        
                        tqdm.write(f"\n🎉 JMultiWOZ ファインチューニング完了！")
                        tqdm.write(f"   総時間: {elapsed_str}")
                        tqdm.write(f"   モデルID: {job.fine_tuned_model}")
                        return job.fine_tuned_model
                    
                    elif job.status == "failed":
                        tqdm.write(f"\n❌ ファインチューニング失敗")
                        if hasattr(job, 'error') and job.error:
                            tqdm.write(f"   エラー: {job.error}")
                        return None
                    
                    elif job.status == "cancelled":
                        tqdm.write(f"\n⚠️  ファインチューニングがキャンセルされました")
                        return None
                    
                    # 学習中の詳細情報表示
                    if job.status == "running":
                        try:
                            events = self.client.fine_tuning.jobs.list_events(
                                fine_tuning_job_id=self.job_id,
                                limit=1
                            )
                            if events.data and events.data[0].message:
                                message = events.data[0].message
                                if "Step" in message or "Epoch" in message:
                                    tqdm.write(f"   📈 {message}")
                        except:
                            pass
                    
                    time.sleep(10)  # 10秒ごとにチェック
                    
            except KeyboardInterrupt:
                tqdm.write("\n\n⚠️  監視を中止しました")
                tqdm.write(f"   ジョブは継続中です: {self.job_id}")
                return None
    
    def test_jmultiwoz_model(self, model_id):
        """JMultiWOZファインチューニング済みモデルをテスト"""
        
        print(f"\n🧪 JMultiWOZファインチューニング済みモデルテスト")
        print(f"   モデルID: {model_id}")
        
        # JMultiWOZドメインのテストプロンプト
        test_prompts = [
            "こんにちは。東京で美味しいレストランを探しています。",
            "京都のホテルを予約したいのですが、おすすめはありますか？",
            "大阪駅から関西空港までの行き方を教えてください。",
            "Wi-Fi完備で価格が手頃なカフェを教えてください。",
            "今度の週末に家族でショッピングモールに行きたいです。"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n【JMultiWOZ テスト {i}/{len(test_prompts)}】")
            print(f"👤 お客様: {prompt}")
            
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "system", 
                            "content": "あなたは親切で知識豊富なカスタマーサービスの担当者です。お客様の質問や要求に丁寧に対応してください。"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                
                print(f"🤖 スタッフ: {response.choices[0].message.content}")
                
            except Exception as e:
                print(f"❌ エラー: {e}")

def main():
    """メイン実行関数"""
    
    print("=" * 60)
    print("🎯 JMultiWOZ ファインチューニング実行")
    print("=" * 60)
    
    # 修正済みデータファイルの確認
    train_file = Path("data/jmultiwoz_train_fixed.jsonl")
    val_file = Path("data/jmultiwoz_validation_fixed.jsonl")
    
    if not train_file.exists() or not val_file.exists():
        print("❌ JMultiWOZデータファイルが見つかりません")
        print("先に prepare_jmultiwoz.py を実行してください")
        return
    
    print(f"📊 データ確認:")
    print(f"  トレーニング: {train_file}")
    print(f"  バリデーション: {val_file}")
    
    # ファインチューナーの初期化
    tuner = JMultiWOZFineTuner()
    
    # JMultiWOZデータのアップロード
    tuner.upload_jmultiwoz_data(train_file, val_file)
    
    # ファインチューニング開始
    job_id = tuner.start_jmultiwoz_finetuning()
    
    if job_id:
        # 学習の監視
        model_id = tuner.monitor_jmultiwoz_training()
        
        if model_id:
            # モデルのテスト
            tuner.test_jmultiwoz_model(model_id)
            
            print("\n" + "=" * 60)
            print("🎉 JMultiWOZ ファインチューニング完了！")
            print(f"   学習データ: 日本語タスク指向対話 (700+300件)")
            print(f"   ファインチューニング済みモデル: {model_id}")
            print(f"   適用分野: カスタマーサービス、予約システム、案内業務")
            print("\n使用方法:")
            print(f'   model="{model_id}"')
            print("=" * 60)
        else:
            print("\n⚠️  ファインチューニングが完了していません")
            print(f"   ジョブID: {job_id}")
            print("   後で確認してください")
    else:
        print("\n❌ ファインチューニングを開始できませんでした")

if __name__ == "__main__":
    main()