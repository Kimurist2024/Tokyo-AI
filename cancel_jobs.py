"""
GPT-5 nano以外のファインチューニングジョブをキャンセル
"""

from api_key_manager import APIKeyManager
from openai import OpenAI

def cancel_non_gpt5_jobs():
    """GPT-5 nano以外のジョブをキャンセル"""
    
    manager = APIKeyManager()
    client = OpenAI(api_key=manager.get_key("OPENAI_API_KEY"))
    
    print("🔍 現在のファインチューニングジョブを確認...")
    
    # 全ジョブを取得
    jobs = client.fine_tuning.jobs.list()
    
    jobs_to_cancel = []
    gpt5_jobs = []
    
    for job in jobs.data:
        print(f"ID: {job.id} | Status: {job.status} | Model: {job.model}")
        
        if job.status in ["validating_files", "queued", "running"]:  # キャンセル可能な状態
            if "gpt-5" in job.model.lower():
                gpt5_jobs.append(job)
                print(f"  ✅ GPT-5関連 - 保持します")
            else:
                jobs_to_cancel.append(job)
                print(f"  ❌ GPT-5以外 - キャンセル対象")
    
    if not jobs_to_cancel:
        print("\n✅ キャンセル対象のジョブはありません")
        return
    
    print(f"\n⚠️  {len(jobs_to_cancel)}個のジョブをキャンセルします:")
    for job in jobs_to_cancel:
        print(f"  - {job.id} ({job.model})")
    
    # 確認
    confirm = input("\nキャンセルしますか？ (y/N): ").strip().lower()
    
    if confirm == 'y':
        print("\n🗑️  ジョブをキャンセル中...")
        
        for job in jobs_to_cancel:
            try:
                client.fine_tuning.jobs.cancel(job.id)
                print(f"  ✅ キャンセル: {job.id}")
            except Exception as e:
                print(f"  ❌ エラー {job.id}: {e}")
        
        print("\n✅ キャンセル完了")
    else:
        print("\n⚠️  キャンセルを中止しました")


if __name__ == "__main__":
    cancel_non_gpt5_jobs()