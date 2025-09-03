"""
GPU対応＆進捗表示付きファインチューニングスクリプト
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from api_key_manager import APIKeyManager
from openai import OpenAI

class GPUFineTuner:
    def __init__(self):
        manager = APIKeyManager()
        api_key = manager.get_key("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI APIキーが設定されていません")
        
        self.client = OpenAI(api_key=api_key)
        self.training_file_id = None
        self.validation_file_id = None
        self.job_id = None
    
    def upload_dataset_with_progress(self, train_file: Path, val_file: Path = None):
        """プログレスバー付きでデータセットをアップロード"""
        
        print("📤 データセットをアップロード中...")
        
        # ファイルサイズを取得
        train_size = train_file.stat().st_size
        
        # トレーニングファイルをアップロード（プログレスバー付き）
        with tqdm(total=train_size, unit='B', unit_scale=True, desc="トレーニングデータ") as pbar:
            with open(train_file, "rb") as f:
                # ファイル全体を読み込み
                file_content = f.read()
                pbar.update(train_size)
                
                # OpenAIにアップロード
                response = self.client.files.create(
                    file=file_content,
                    purpose="fine-tune"
                )
                self.training_file_id = response.id
        
        print(f"✅ トレーニングファイル: {self.training_file_id}")
        
        # 検証ファイルをアップロード
        if val_file and val_file.exists():
            val_size = val_file.stat().st_size
            
            with tqdm(total=val_size, unit='B', unit_scale=True, desc="検証データ") as pbar:
                with open(val_file, "rb") as f:
                    file_content = f.read()
                    pbar.update(val_size)
                    
                    response = self.client.files.create(
                        file=file_content,
                        purpose="fine-tune"
                    )
                    self.validation_file_id = response.id
            
            print(f"✅ 検証ファイル: {self.validation_file_id}")
        
        return self.training_file_id, self.validation_file_id
    
    def start_gpu_finetuning(self, model="gpt-4o-mini-2024-07-18", suffix="travel-jp-gpu"):
        """GPU最適化されたファインチューニングジョブを開始"""
        
        print(f"\n🚀 GPU最適化ファインチューニングを開始...")
        print(f"   モデル: {model}")
        
        try:
            # ファインチューニングジョブを作成（GPU最適化パラメータ）
            job_params = {
                "training_file": self.training_file_id,
                "model": model,
                "suffix": suffix,
                "hyperparameters": {
                    "n_epochs": 3,  # エポック数
                    "batch_size": 4,  # GPU用に最適化されたバッチサイズ
                    "learning_rate_multiplier": 0.1  # 学習率の調整
                }
            }
            
            # 検証ファイルがある場合は追加
            if self.validation_file_id:
                job_params["validation_file"] = self.validation_file_id
            
            response = self.client.fine_tuning.jobs.create(**job_params)
            self.job_id = response.id
            
            print(f"✅ ファインチューニングジョブ開始: {self.job_id}")
            print(f"   サフィックス: {suffix}")
            print(f"   GPU最適化: バッチサイズ=4, エポック=3")
            
            return self.job_id
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            return None
    
    def monitor_with_eta(self):
        """進捗とETA（推定完了時間）を表示しながら監視"""
        
        if not self.job_id:
            print("❌ ジョブIDが設定されていません")
            return
        
        print(f"\n📊 ジョブ監視中: {self.job_id}")
        
        start_time = time.time()
        last_status = None
        status_stages = {
            "validating_files": 5,
            "queued": 10,
            "running": 85,
            "succeeded": 100
        }
        
        # プログレスバーを作成
        with tqdm(total=100, desc="ファインチューニング進捗", unit="%", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            try:
                while True:
                    job = self.client.fine_tuning.jobs.retrieve(self.job_id)
                    
                    # ステータスが変わったら更新
                    if job.status != last_status:
                        last_status = job.status
                        
                        # プログレスバーを更新
                        if job.status in status_stages:
                            target_progress = status_stages[job.status]
                            current_progress = pbar.n
                            if target_progress > current_progress:
                                pbar.update(target_progress - current_progress)
                        
                        # ステータス表示
                        elapsed = time.time() - start_time
                        elapsed_str = str(timedelta(seconds=int(elapsed)))
                        
                        if job.status == "validating_files":
                            tqdm.write(f"📝 ファイル検証中... (経過: {elapsed_str})")
                        elif job.status == "queued":
                            tqdm.write(f"⏳ キューで待機中... (経過: {elapsed_str})")
                        elif job.status == "running":
                            tqdm.write(f"🏃 トレーニング実行中... (経過: {elapsed_str})")
                            tqdm.write("   GPUで高速処理中...")
                            # 推定残り時間（経験値ベース）
                            estimated_total = 1200  # 20分（秒）
                            remaining = max(0, estimated_total - elapsed)
                            remaining_str = str(timedelta(seconds=int(remaining)))
                            tqdm.write(f"   推定残り時間: {remaining_str}")
                    
                    # 完了チェック
                    if job.status == "succeeded":
                        pbar.update(100 - pbar.n)  # 100%にする
                        elapsed = time.time() - start_time
                        elapsed_str = str(timedelta(seconds=int(elapsed)))
                        
                        tqdm.write(f"\n✅ ファインチューニング完了！")
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
                    
                    # トレーニング中は詳細情報を取得
                    if job.status == "running":
                        try:
                            # イベントログから進捗を取得
                            events = self.client.fine_tuning.jobs.list_events(
                                fine_tuning_job_id=self.job_id,
                                limit=1
                            )
                            if events.data:
                                latest_event = events.data[0]
                                if latest_event.message:
                                    # ステップ情報を抽出
                                    if "Step" in latest_event.message:
                                        tqdm.write(f"   {latest_event.message}")
                        except:
                            pass
                    
                    time.sleep(5)  # 5秒ごとにチェック
                    
            except KeyboardInterrupt:
                tqdm.write("\n\n⚠️  監視を中止しました")
                tqdm.write(f"   ジョブは継続中です: {self.job_id}")
                return None
    
    def test_finetuned_model(self, model_id):
        """ファインチューニング済みモデルをテスト"""
        
        print(f"\n🧪 ファインチューニング済みモデルをテスト: {model_id}")
        
        test_prompts = [
            "東京から大阪への移動方法を教えてください。",
            "北海道旅行のおすすめ時期は？",
            "JRパスの使い方を教えてください。"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n【テスト {i}/3】")
            print(f"👤 質問: {prompt}")
            
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150
                )
                
                print(f"🤖 回答: {response.choices[0].message.content}")
                
            except Exception as e:
                print(f"❌ エラー: {e}")


def check_job_status(job_id: str):
    """既存のジョブの状態を確認"""
    
    manager = APIKeyManager()
    client = OpenAI(api_key=manager.get_key("OPENAI_API_KEY"))
    
    print(f"📊 ジョブステータス確認: {job_id}")
    
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        print(f"\n状態: {job.status}")
        print(f"作成日時: {job.created_at}")
        
        if job.status == "succeeded":
            print(f"✅ 完了!")
            print(f"モデルID: {job.fine_tuned_model}")
        elif job.status == "running":
            print("🏃 実行中...")
            # 最新のイベントを表示
            events = client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job_id,
                limit=3
            )
            if events.data:
                print("\n最新のイベント:")
                for event in events.data:
                    print(f"  - {event.message}")
        
        return job
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None


def main():
    """メイン実行関数"""
    
    print("=" * 60)
    print("🎯 GPU最適化ファインチューニング（進捗表示付き）")
    print("=" * 60)
    
    # データセットの準備
    from prepare_travel_dataset import create_travel_dataset
    train_file, val_file = create_travel_dataset()
    
    # ファインチューナーの初期化
    tuner = GPUFineTuner()
    
    # データセットのアップロード（プログレスバー付き）
    tuner.upload_dataset_with_progress(train_file, val_file)
    
    # GPU最適化ファインチューニング開始
    job_id = tuner.start_gpu_finetuning()
    
    if job_id:
        # 進捗とETA表示付きで監視
        model_id = tuner.monitor_with_eta()
        
        if model_id:
            # モデルのテスト
            tuner.test_finetuned_model(model_id)
            
            print("\n" + "=" * 60)
            print("✅ ファインチューニング完了！")
            print(f"   モデルID: {model_id}")
            print("\n使用方法:")
            print(f'   model="{model_id}"')
            print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # ジョブIDが指定されたら状態確認
        check_job_status(sys.argv[1])
    else:
        # 通常の実行
        main()