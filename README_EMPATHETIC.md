# Empathetic Llama 3 - 共感的なLLMファインチューニング

Facebook の Empathetic Dialogues データセットを使用して、Llama 3 を共感的な対話ができるようにファインチューニングするプロジェクト。

## 必要な環境

- Python 3.8+
- CUDA対応GPU (推奨: 24GB以上のVRAM)
- PyTorch 2.0+

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. データセット準備

```bash
python prepare_empathetic_dataset.py
```

これにより以下のファイルが生成されます：
- `empathetic_train.json` - 訓練用データ (10,000サンプル)
- `empathetic_val.json` - 検証用データ (1,000サンプル)

### 2. ファインチューニング実行

```bash
python finetune_llama3_empathetic.py
```

設定:
- ベースモデル: `meta-llama/Meta-Llama-3-8B-Instruct`
- 量子化: 4bit (bitsandbytes)
- LoRAを使用したパラメータ効率的学習
- バッチサイズ: 2 (勾配蓄積: 4)
- エポック: 3

### 3. チャットボットとして使用

```bash
# ベースモデルのみでテスト
python empathetic_chat.py --test

# ファインチューニング済みモデルで対話
python empathetic_chat.py --adapter_path ./empathetic-llama3
```

## ファイル構成

- `prepare_empathetic_dataset.py` - データセット前処理
- `finetune_llama3_empathetic.py` - ファインチューニングスクリプト
- `empathetic_chat.py` - 推論・対話用スクリプト
- `requirements.txt` - 必要ライブラリ

## データセットについて

[Facebook Empathetic Dialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues) は、共感的な応答を学習するためのデータセット：

- 25,000の対話
- 32の感情カテゴリ
- Context（状況説明）付きの対話

## ファインチューニング詳細

- **LoRA設定**: r=16, alpha=32, dropout=0.1
- **対象モジュール**: attention層とFeed-Forward層
- **最大シーケンス長**: 1024トークン
- **学習率**: 2e-4
- **オプティマイザー**: AdamW (paged_adamw_32bit)

## 期待される改善

ファインチューニング後のモデルは以下の特徴を持つ：

1. **共感的応答**: ユーザーの感情に寄り添う返答
2. **支持的態度**: 励ましや理解を示す表現
3. **適切なトーン**: 状況に応じた適切な口調

## トラブルシューティング

### CUDA Out of Memory
- バッチサイズを1に減らす
- 勾配蓄積ステップを増やす
- より小さなモデル（Llama 3-8B → 7B）を使用

### HuggingFace認証エラー
```bash
huggingface-cli login
```

### 権限エラー
Llama 3モデルを使用するにはMeta/Hugging Faceでの申請が必要です。

## 性能評価

以下のテストケースで応答品質を確認：

```python
test_cases = [
    "今日はとても疲れました...",
    "友達と喧嘩してしまって悲しいです", 
    "仕事が上手くいかなくて落ち込んでいます"
]
```