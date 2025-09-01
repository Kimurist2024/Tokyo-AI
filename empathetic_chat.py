import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

class EmpatheticChatBot:
    def __init__(self, base_model_path: str, adapter_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if adapter_path:
            print(f"Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                torch_dtype=torch.bfloat16
            )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def create_prompt(self, user_input: str) -> str:
        """共感的なプロンプトを作成"""
        system_prompt = "You are an empathetic assistant who provides supportive and understanding responses to help people feel heard and cared for."
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return prompt
    
    def generate_response(self, user_input: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """応答を生成"""
        prompt = self.create_prompt(user_input)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        assistant_start = full_response.find("<|start_header_id|>assistant<|end_header_id|>")
        if assistant_start != -1:
            response = full_response[assistant_start + len("<|start_header_id|>assistant<|end_header_id|>"):].strip()
        else:
            response = full_response.replace(prompt, "").strip()
        
        return response
    
    def chat_loop(self):
        """対話ループ"""
        print("\n" + "="*50)
        print("Empathetic ChatBot - 共感的なAIチャットボット")
        print("Type 'quit' or 'exit' to end the conversation")
        print("="*50 + "\n")
        
        while True:
            user_input = input("あなた: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("チャットを終了します。お疲れさまでした！")
                break
            
            if not user_input.strip():
                continue
            
            print("ChatBot: ", end="", flush=True)
            
            try:
                response = self.generate_response(user_input)
                print(response)
            except Exception as e:
                print(f"エラーが発生しました: {e}")
            
            print()

def test_responses():
    """テスト用の応答例"""
    test_cases = [
        "今日はとても疲れました...",
        "友達と喧嘩してしまって悲しいです",
        "仕事が上手くいかなくて落ち込んでいます",
        "最近寝れなくて辛いです",
        "試験に落ちてしまいました"
    ]
    
    chatbot = EmpatheticChatBot("meta-llama/Meta-Llama-3-8B-Instruct")
    
    print("\n" + "="*50)
    print("Test Responses:")
    print("="*50)
    
    for test_input in test_cases:
        print(f"\nユーザー: {test_input}")
        response = chatbot.generate_response(test_input)
        print(f"ChatBot: {response}")
        print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="Empathetic ChatBot")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct", 
                       help="Base model path")
    parser.add_argument("--adapter_path", default=None, 
                       help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--test", action="store_true", 
                       help="Run test responses")
    
    args = parser.parse_args()
    
    if args.test:
        test_responses()
    else:
        chatbot = EmpatheticChatBot(args.base_model, args.adapter_path)
        chatbot.chat_loop()

if __name__ == "__main__":
    main()