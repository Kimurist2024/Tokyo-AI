"""
JMultiWOZデータセットをファインチューニング用に変換
7:3でtrain/validationに分割
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

def load_jmultiwoz_data(json_path: str) -> Dict:
    """JMultiWOZデータを読み込み"""
    print(f"📖 JMultiWOZデータを読み込み中: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ {len(data)}件の対話データを読み込みました")
    return data

def convert_dialogue_to_messages(dialogue: Dict) -> List[Dict]:
    """JMultiWOZ対話を OpenAI形式のメッセージに変換"""
    
    messages = []
    
    # システムメッセージを追加
    system_msg = {
        "role": "system",
        "content": "あなたは親切で知識豊富なカスタマーサービスの担当者です。お客様の質問や要求に丁寧に対応してください。"
    }
    messages.append(system_msg)
    
    # 対話の各ターンを変換
    if "turns" in dialogue:
        for turn in dialogue["turns"]:
            if turn.get("speaker") == "USER":
                # ユーザーの発言
                user_msg = {
                    "role": "user", 
                    "content": turn.get("utterance", "").strip()
                }
                if user_msg["content"]:
                    messages.append(user_msg)
            
            elif turn.get("speaker") == "SYSTEM":
                # システム（アシスタント）の発言
                assistant_msg = {
                    "role": "assistant",
                    "content": turn.get("utterance", "").strip()
                }
                if assistant_msg["content"]:
                    messages.append(assistant_msg)
    
    return messages

def process_jmultiwoz_for_finetuning(data: Dict, max_dialogues: int = 1000) -> List[Dict]:
    """JMultiWOZデータをファインチューニング用に処理"""
    
    print(f"🔄 {min(max_dialogues, len(data))}件の対話を処理中...")
    
    processed_data = []
    dialogue_keys = list(data.keys())[:max_dialogues]  # 最大件数制限
    
    for dialogue_id in tqdm(dialogue_keys, desc="対話処理"):
        dialogue = data[dialogue_id]
        
        # 対話をメッセージ形式に変換
        messages = convert_dialogue_to_messages(dialogue)
        
        # 最低限のメッセージ数をチェック（システム + ユーザー + アシスタント）
        if len(messages) >= 3:
            training_example = {"messages": messages}
            processed_data.append(training_example)
    
    print(f"✅ {len(processed_data)}件の学習用データを作成しました")
    return processed_data

def split_train_validation(data: List[Dict], train_ratio: float = 0.7) -> Tuple[List[Dict], List[Dict]]:
    """データを7:3でtrain/validationに分割"""
    
    print(f"📊 データを{int(train_ratio*100)}:{int((1-train_ratio)*100)}で分割中...")
    
    # データをシャッフル
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 分割点を計算
    split_index = int(len(shuffled_data) * train_ratio)
    
    train_data = shuffled_data[:split_index]
    validation_data = shuffled_data[split_index:]
    
    print(f"✅ トレーニング: {len(train_data)}件")
    print(f"✅ バリデーション: {len(validation_data)}件")
    
    return train_data, validation_data

def save_jsonl(data: List[Dict], file_path: Path):
    """データをJSONL形式で保存"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"💾 保存完了: {file_path} ({len(data)}件)")

def validate_dataset(file_path: Path) -> bool:
    """データセットの形式を検証"""
    
    print(f"🔍 データセット検証: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # 最初の3行だけチェック
                    break
                    
                data = json.loads(line)
                
                # 必須フィールドの確認
                assert "messages" in data, "messages フィールドが必要です"
                assert len(data["messages"]) >= 2, "最低2つのメッセージが必要です"
                
                # 各メッセージの検証
                for msg in data["messages"]:
                    assert "role" in msg, "role フィールドが必要です"
                    assert "content" in msg, "content フィールドが必要です"
                    assert msg["role"] in ["system", "user", "assistant"], f"不正なrole: {msg['role']}"
        
        print("  ✅ 検証成功！")
        return True
        
    except Exception as e:
        print(f"  ❌ 検証失敗: {e}")
        return False

def show_sample_data(data: List[Dict], num_samples: int = 2):
    """サンプルデータを表示"""
    
    print(f"\n📝 データサンプル（{num_samples}件）:")
    
    for i, sample in enumerate(data[:num_samples]):
        print(f"\n--- サンプル {i+1} ---")
        for j, message in enumerate(sample["messages"][:4]):  # 最初の4メッセージのみ
            role = message["role"]
            content = message["content"][:100]  # 最初の100文字のみ
            print(f"  {j+1}. {role}: {content}...")
        
        if len(sample["messages"]) > 4:
            print(f"  ... 他 {len(sample['messages']) - 4} メッセージ")

def main():
    """メイン処理"""
    
    print("=" * 60)
    print("🎯 JMultiWOZ ファインチューニングデータ準備")
    print("=" * 60)
    
    # JMultiWOZデータの読み込み
    jmultiwoz_path = "/root/Tokyo-AI/jmultiwoz/dataset/JMultiWOZ_1.0/dialogues.json"
    data = load_jmultiwoz_data(jmultiwoz_path)
    
    # ファインチューニング用に処理
    processed_data = process_jmultiwoz_for_finetuning(data, max_dialogues=1000)
    
    if not processed_data:
        print("❌ 処理できるデータがありませんでした")
        return
    
    # サンプルデータ表示
    show_sample_data(processed_data)
    
    # 7:3で分割
    train_data, validation_data = split_train_validation(processed_data, train_ratio=0.7)
    
    # 出力ディレクトリ作成
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # JSONLファイルとして保存
    train_file = output_dir / "jmultiwoz_train.jsonl"
    val_file = output_dir / "jmultiwoz_validation.jsonl"
    
    save_jsonl(train_data, train_file)
    save_jsonl(validation_data, val_file)
    
    # データセット検証
    print(f"\n🔍 データセット検証中...")
    train_valid = validate_dataset(train_file)
    val_valid = validate_dataset(val_file)
    
    if train_valid and val_valid:
        print("\n" + "=" * 60)
        print("✅ JMultiWOZ ファインチューニングデータ準備完了！")
        print(f"  トレーニング: {train_file} ({len(train_data)}件)")
        print(f"  バリデーション: {val_file} ({len(validation_data)}件)")
        print("\n次のステップ:")
        print("  python finetune_jmultiwoz.py")
        print("=" * 60)
        
        return train_file, val_file
    else:
        print("❌ データセット検証に失敗しました")
        return None, None

if __name__ == "__main__":
    # シード固定で再現性を確保
    random.seed(42)
    main()