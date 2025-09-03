"""
GPT-5でファインチューニングを実行
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from api_key_manager import APIKeyManager
from openai import OpenAI

class GPT5FineTuner:
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
                response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
                self.training_file_id = response.id
                pbar.update(train_size)
        
        print(f"✅ トレーニングファイル: {self.training_file_id}")
        
        # 検証ファイルをアップロード
        if val_file and val_file.exists():
            val_size = val_file.stat().st_size
            
            with tqdm(total=val_size, unit='B', unit_scale=True, desc="検証データ") as pbar:
                with open(val_file, "rb") as f:
                    response = self.client.files.create(
                        file=f,
                        purpose="fine-tune"
                    )
                    self.validation_file_id = response.id
                    pbar.update(val_size)
            
            print(f"✅ 検証ファイル: {self.validation_file_id}")
        
        return self.training_file_id, self.validation_file_id
    
    def try_gpt5_models(self, suffix="travel-jp-gpt5"):
        """GPT-5系モデルでファインチューニングを試行"""
        
        # GPT-5系モデルのリスト（優先順）
        gpt5_models = [
            "gpt-5",
            "gpt-5-2025-08-07",
            "gpt-5-nano",
            "gpt-5-nano-2025-08-07",
            "gpt-5-mini",
            "gpt-5-mini-2025-08-07"
        ]
        
        for model in gpt5_models:
            print(f"\n🎯 {model} でファインチューニングを試行...")
            
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
                
                # GPT-5用のハイパーパラメータ（推測）
                job_params["hyperparameters"] = {
                    "n_epochs": 2,  # GPT-5は効率が良いかもしれないので少なめ
                    "learning_rate_multiplier": 0.05  # より低い学習率
                }
                
                response = self.client.fine_tuning.jobs.create(**job_params)
                self.job_id = response.id
                
                print(f"✅ ファインチューニングジョブ開始: {self.job_id}")
                print(f"   モデル: {model}")
                print(f"   サフィックス: {suffix}")
                
                return self.job_id, model
                
            except Exception as e:
                error_str = str(e)
                print(f"❌ {model} でエラー: {error_str}")
                
                if "not available" in error_str.lower() or "does not exist" in error_str.lower():
                    print(f"   → {model} はファインチューニングに対応していません")
                    continue
                else:
                    print(f"   → 予期しないエラー: {e}")
                    continue
        
        # 全てのGPT-5モデルが失敗した場合
        print("\n⚠️  すべてのGPT-5モデルでファインチューニングが失敗しました")
        print("\n💡 代替案として gpt-4o または gpt-4o-mini を使用してください")
        return None, None
    
    def monitor_with_eta(self, model_name):
        """進捗とETA表示付きで監視"""
        
        if not self.job_id:
            print("❌ ジョブIDが設定されていません")
            return
        
        print(f"\n📊 ジョブ監視中: {self.job_id} ({model_name})")
        
        start_time = time.time()
        last_status = None
        status_stages = {
            "validating_files": 5,
            "queued": 15,
            "running": 85,
            "succeeded": 100
        }
        
        # プログレスバーを作成
        with tqdm(total=100, desc=f"GPT-5 ファインチューニング", unit="%", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
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
                            tqdm.write(f"🚀 GPT-5 トレーニング実行中... (経過: {elapsed_str})")
                            # GPT-5は高性能なので推定時間は短め
                            estimated_total = 600  # 10分（推測）
                            remaining = max(0, estimated_total - elapsed)
                            remaining_str = str(timedelta(seconds=int(remaining)))
                            tqdm.write(f"   推定残り時間: {remaining_str}")
                    
                    # 完了チェック
                    if job.status == "succeeded":
                        pbar.update(100 - pbar.n)  # 100%にする
                        elapsed = time.time() - start_time
                        elapsed_str = str(timedelta(seconds=int(elapsed)))
                        
                        tqdm.write(f"\n🎉 GPT-5 ファインチューニング完了！")
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
                    
                    time.sleep(5)  # 5秒ごとにチェック
                    
            except KeyboardInterrupt:
                tqdm.write("\n\n⚠️  監視を中止しました")
                tqdm.write(f"   ジョブは継続中です: {self.job_id}")
                return None
    
    def test_gpt5_model(self, model_id):
        """GPT-5ファインチューニング済みモデルをテスト"""
        
        print(f"\n🧪 GPT-5ファインチューニング済みモデルをテスト: {model_id}")
        
        test_prompts = [
            "東京から大阪への移動方法を教えてください。",
            "北海道旅行のおすすめ時期は？", 
            "富士山に登るのに必要な装備は？",
            "JRパスの使い方を教えてください。"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n【GPT-5テスト {i}/{len(test_prompts)}】")
            print(f"👤 質問: {prompt}")
            
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "あなたは親切で知識豊富な旅行代理店のエージェントです。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=200
                )
                
                print(f"🤖 GPT-5回答: {response.choices[0].message.content}")
                
            except Exception as e:
                print(f"❌ エラー: {e}")


def main():
    """メイン実行関数"""
    
    print("=" * 60)
    print("🚀 GPT-5 ファインチューニング")
    print("=" * 60)
    
    # データセットの準備
    from prepare_travel_dataset import create_travel_dataset
    train_file, val_file = create_travel_dataset()
    
    # ファインチューナーの初期化
    tuner = GPT5FineTuner()
    
    # データセットのアップロード
    tuner.upload_dataset_with_progress(train_file, val_file)
    
    # GPT-5でファインチューニング開始
    job_id, model_name = tuner.try_gpt5_models()
    
    if job_id and model_name:
        # ジョブの監視
        model_id = tuner.monitor_with_eta(model_name)
        
        if model_id:
            # GPT-5モデルのテスト
            tuner.test_gpt5_model(model_id)
            
            print("\n" + "=" * 60)
            print("🎉 GPT-5 ファインチューニング完了！")
            print(f"   ベースモデル: {model_name}")
            print(f"   ファインチューニング済みモデルID: {model_id}")
            print("\n使用方法:")
            print(f'   model="{model_id}"')
            print("=" * 60)
        else:
            print("\n⚠️  ファインチューニングが完了していません")
            print(f"   ジョブID: {job_id}")
    else:
        print("\n❌ GPT-5でファインチューニングを開始できませんでした")
        print("\n💡 gpt-4o-mini を使用することをおすすめします:")
        print("   python finetune_with_gpu.py")


if __name__ == "__main__":
    main()