"""
JMultiWOZデータの形式エラーを修正
最後のメッセージがassistantで終わるように調整
"""

import json
from pathlib import Path
from tqdm import tqdm

def fix_dialogue_format(messages):
    """対話の形式を修正"""
    
    if not messages or len(messages) < 2:
        return None
    
    # システムメッセージを除いた会話部分を抽出
    conversation = [msg for msg in messages if msg["role"] != "system"]
    
    if not conversation:
        return None
    
    # 最後のメッセージがassistantでない場合は除去
    while conversation and conversation[-1]["role"] != "assistant":
        conversation.pop()
    
    # 対話が短すぎる場合はスキップ
    if len(conversation) < 2:
        return None
    
    # user -> assistant の交互パターンを確保
    fixed_conversation = []
    expected_role = "user"
    
    for msg in conversation:
        if msg["role"] == expected_role:
            fixed_conversation.append(msg)
            expected_role = "assistant" if expected_role == "user" else "user"
    
    # 最終的にassistantで終わることを確認
    if not fixed_conversation or fixed_conversation[-1]["role"] != "assistant":
        return None
    
    # システムメッセージを先頭に追加
    system_msg = {
        "role": "system",
        "content": "あなたは親切で知識豊富なカスタマーサービスの担当者です。お客様の質問や要求に丁寧に対応してください。"
    }
    
    return [system_msg] + fixed_conversation

def fix_jsonl_file(input_file: Path, output_file: Path):
    """JSONLファイルの形式を修正"""
    
    print(f"🔧 修正中: {input_file}")
    
    fixed_data = []
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(tqdm(lines, desc="データ修正")):
        try:
            data = json.loads(line)
            messages = data.get("messages", [])
            
            # 形式を修正
            fixed_messages = fix_dialogue_format(messages)
            
            if fixed_messages:
                fixed_data.append({"messages": fixed_messages})
            else:
                error_count += 1
                
        except Exception as e:
            print(f"行 {i+1} でエラー: {e}")
            error_count += 1
    
    # 修正されたデータを保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in fixed_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ 修正完了: {output_file}")
    print(f"   有効データ: {len(fixed_data)}件")
    print(f"   エラー・除外: {error_count}件")
    
    return len(fixed_data)

def validate_fixed_data(file_path: Path):
    """修正されたデータを検証"""
    
    print(f"🔍 修正データの検証: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # 最初の5行をチェック
                break
                
            data = json.loads(line)
            messages = data["messages"]
            
            # 基本チェック
            assert len(messages) >= 3, f"行 {i+1}: メッセージ数不足"
            assert messages[0]["role"] == "system", f"行 {i+1}: 最初はsystemメッセージ"
            assert messages[-1]["role"] == "assistant", f"行 {i+1}: 最後はassistantメッセージ"
            
            # 交互チェック
            conversation = [msg for msg in messages if msg["role"] != "system"]
            for j in range(len(conversation)):
                expected_role = "user" if j % 2 == 0 else "assistant"
                actual_role = conversation[j]["role"]
                assert actual_role == expected_role, f"行 {i+1}: 役割の順序エラー"
    
    print("  ✅ 検証成功！")

def show_sample(file_path: Path, num_samples: int = 2):
    """サンプルデータを表示"""
    
    print(f"\n📝 修正後のサンプル（{num_samples}件）:")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
                
            data = json.loads(line)
            messages = data["messages"]
            
            print(f"\n--- サンプル {i+1} ---")
            for j, msg in enumerate(messages[:4]):  # 最初の4メッセージ
                role = msg["role"]
                content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                print(f"  {j+1}. {role}: {content}")
            
            if len(messages) > 4:
                print(f"  ... 他 {len(messages) - 4} メッセージ")
                print(f"  最終: {messages[-1]['role']}: {messages[-1]['content'][:80]}...")

def main():
    """メイン処理"""
    
    print("=" * 60)
    print("🔧 JMultiWOZ データ形式修正")
    print("=" * 60)
    
    # 入力ファイル
    train_input = Path("data/jmultiwoz_train.jsonl")
    val_input = Path("data/jmultiwoz_validation.jsonl")
    
    # 出力ファイル
    train_output = Path("data/jmultiwoz_train_fixed.jsonl")
    val_output = Path("data/jmultiwoz_validation_fixed.jsonl")
    
    if not train_input.exists() or not val_input.exists():
        print("❌ 元データファイルが見つかりません")
        return
    
    # トレーニングデータを修正
    train_count = fix_jsonl_file(train_input, train_output)
    
    # バリデーションデータを修正
    val_count = fix_jsonl_file(val_input, val_output)
    
    # 修正されたデータを検証
    validate_fixed_data(train_output)
    validate_fixed_data(val_output)
    
    # サンプル表示
    show_sample(train_output)
    
    print("\n" + "=" * 60)
    print("✅ JMultiWOZ データ修正完了！")
    print(f"   修正済みトレーニング: {train_output} ({train_count}件)")
    print(f"   修正済みバリデーション: {val_output} ({val_count}件)")
    print("\n次のステップ:")
    print("   python finetune_jmultiwoz_fixed.py")
    print("=" * 60)

if __name__ == "__main__":
    main()