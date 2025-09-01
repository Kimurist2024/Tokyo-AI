# AIチャット実装ガイド

## 推奨サービス比較

| サービス | 無料枠 | 速度 | 日本語 | 特徴 |
|---------|--------|------|--------|------|
| **Groq** | ✅ あり | ⚡超高速 | ○ | 最速、Llama3対応 |
| **Gemini** | ✅ 60req/分 | 速い | ◎ | Google製、無料枠豊富 |
| **Ollama** | ✅ 完全無料 | 普通 | ○ | ローカル実行、プライバシー◎ |
| **Claude** | ❌ | 速い | ◎ | 高品質、日本語最強 |
| **GPT-4** | ❌ | 普通 | ◎ | 最高性能、高価 |

## セットアップ方法

### 1. Groq (推奨: 高速・安価)
```bash
# APIキー取得
# https://console.groq.com でアカウント作成

# 必要ライブラリ
pip install groq python-dotenv

# 実行
export GROQ_API_KEY='your-api-key'
python groq_example.py
```

### 2. Ollama (完全無料・ローカル)
```bash
# インストール
curl -fsSL https://ollama.com/install.sh | sh

# モデル取得
ollama pull llama3

# サーバー起動
ollama serve

# 別ターミナルで実行
pip install requests
python ollama_example.py
```

### 3. Google Gemini (無料枠豊富)
```bash
# APIキー取得
# https://makersuite.google.com/app/apikey

# 必要ライブラリ
pip install google-generativeai

# 実行
export GEMINI_API_KEY='your-api-key'
python gemini_example.py
```

## 用途別おすすめ

- **商用アプリ**: Groq or Gemini (コスパ最強)
- **プライバシー重視**: Ollama (データがローカル)
- **高品質な日本語**: Claude API
- **プロトタイプ**: Gemini (無料枠で十分)
- **最高性能**: GPT-4 (予算があれば)

## コスト目安（1000リクエスト）

- Ollama: **$0** (サーバー代のみ)
- Gemini: **$0** (無料枠内)
- Groq: **$0.05～0.10**
- Claude: **$0.80～3.00**
- GPT-4: **$10～30**