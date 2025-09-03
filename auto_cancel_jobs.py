"""
GPT-5 nano以外のファインチューニングジョブを自動キャンセル
"""

from api_key_manager import APIKeyManager
from openai import OpenAI

def auto_cancel_non_gpt5_jobs():
    """GPT-5 nano以外のジョブを自動キャンセル"""
    
    manager = APIKeyManager()
    client = OpenAI(api_key=manager.get_key("OPENAI_API_KEY"))
    
    print("🔍 現在のファインチューニングジョブを確認...")
    
    # 全ジョブを取得
    jobs = client.fine_tuning.jobs.list()
    
    jobs_to_cancel = []
    
    for job in jobs.data:
        print(f"ID: {job.id} | Status: {job.status} | Model: {job.model}")
        
        if job.status in ["validating_files", "queued", "running"]:  # キャンセル可能な状態
            if "gpt-5" not in job.model.lower():  # GPT-5以外
                jobs_to_cancel.append(job)
                print(f"  ❌ GPT-5以外 - キャンセル対象")
            else:
                print(f"  ✅ GPT-5関連 - 保持")
    
    if not jobs_to_cancel:
        print("\n✅ キャンセル対象のジョブはありません")
        return
    
    print(f"\n🗑️  {len(jobs_to_cancel)}個のジョブを自動キャンセル中...")
    
    cancelled_count = 0
    for job in jobs_to_cancel:
        try:
            client.fine_tuning.jobs.cancel(job.id)
            print(f"  ✅ キャンセル: {job.id}")
            cancelled_count += 1
        except Exception as e:
            print(f"  ❌ エラー {job.id}: {e}")
    
    print(f"\n✅ {cancelled_count}個のジョブをキャンセルしました")
    
    # キャンセル後の状態を確認
    print("\n📊 キャンセル後の状態:")
    jobs = client.fine_tuning.jobs.list()
    for job in jobs.data[:5]:  # 最新5件を表示
        print(f"ID: {job.id} | Status: {job.status} | Model: {job.model}")


if __name__ == "__main__":
    auto_cancel_non_gpt5_jobs()