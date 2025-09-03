import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

class APIKeyManager:
    """APIキーを安全に管理するクラス"""
    
    def __init__(self, env_file: str = ".env"):
        """
        APIキーマネージャーの初期化
        
        Args:
            env_file: 環境変数ファイルのパス（デフォルト: .env）
        """
        self.env_file = Path(env_file)
        self.load_keys()
    
    def load_keys(self) -> None:
        """環境変数ファイルからAPIキーを読み込む"""
        if self.env_file.exists():
            load_dotenv(self.env_file)
            print(f"✅ 環境変数を {self.env_file} から読み込みました")
        else:
            print(f"⚠️  {self.env_file} が見つかりません")
            self.create_env_file()
    
    def create_env_file(self) -> None:
        """環境変数ファイルのテンプレートを作成"""
        template = """# APIキーの設定
# 警告: このファイルは絶対にGitにコミットしないでください！

# OpenAI API Key
OPENAI_API_KEY=

# Claude API Key (Anthropic)
ANTHROPIC_API_KEY=

# Google Gemini API Key
GOOGLE_API_KEY=

# Hugging Face API Token
HUGGINGFACE_TOKEN=
"""
        self.env_file.write_text(template)
        print(f"📝 {self.env_file} を作成しました。APIキーを入力してください。")
    
    def get_key(self, key_name: str) -> Optional[str]:
        """
        指定されたAPIキーを取得
        
        Args:
            key_name: 環境変数名（例: OPENAI_API_KEY）
        
        Returns:
            APIキーの値、または存在しない場合はNone
        """
        key = os.getenv(key_name)
        if not key or key.strip() == "":
            print(f"⚠️  {key_name} が設定されていません")
            return None
        return key
    
    def set_key(self, key_name: str, key_value: str) -> None:
        """
        APIキーを環境変数ファイルに保存
        
        Args:
            key_name: 環境変数名
            key_value: APIキーの値
        """
        # 現在の内容を読み込む
        if self.env_file.exists():
            content = self.env_file.read_text()
        else:
            content = ""
        
        # キーが既に存在するか確認
        lines = content.split('\n')
        key_found = False
        
        for i, line in enumerate(lines):
            if line.startswith(f"{key_name}="):
                lines[i] = f"{key_name}={key_value}"
                key_found = True
                break
        
        # キーが存在しない場合は追加
        if not key_found:
            lines.append(f"{key_name}={key_value}")
        
        # ファイルに書き込む
        self.env_file.write_text('\n'.join(lines))
        
        # 環境変数も更新
        os.environ[key_name] = key_value
        print(f"✅ {key_name} を保存しました")
    
    def list_keys(self) -> dict:
        """
        設定されているすべてのAPIキーをリスト表示
        
        Returns:
            キー名と設定状態の辞書
        """
        keys = {
            "OPENAI_API_KEY": "OpenAI",
            "ANTHROPIC_API_KEY": "Claude (Anthropic)",
            "GOOGLE_API_KEY": "Google Gemini",
            "HUGGINGFACE_TOKEN": "Hugging Face"
        }
        
        status = {}
        for key, service in keys.items():
            value = self.get_key(key)
            if value:
                # キーの一部だけ表示（セキュリティのため）
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                status[service] = f"設定済み ({masked})"
            else:
                status[service] = "未設定"
        
        return status
    
    def validate_openai_key(self) -> bool:
        """OpenAI APIキーの検証"""
        key = self.get_key("OPENAI_API_KEY")
        if not key:
            return False
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            # 簡単なテストリクエスト
            client.models.list()
            print("✅ OpenAI APIキーは有効です")
            return True
        except Exception as e:
            print(f"❌ OpenAI APIキーの検証に失敗: {e}")
            return False


if __name__ == "__main__":
    # 使用例
    manager = APIKeyManager()
    
    print("\n📋 APIキーの状態:")
    for service, status in manager.list_keys().items():
        print(f"  {service}: {status}")
    
    print("\n💡 使用方法:")
    print("  1. .envファイルを開いてAPIキーを入力")
    print("  2. またはset_key()メソッドを使用:")
    print('     manager.set_key("OPENAI_API_KEY", "your-key-here")')
    print("  3. get_key()でAPIキーを取得:")
    print('     key = manager.get_key("OPENAI_API_KEY")')