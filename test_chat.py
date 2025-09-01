#!/usr/bin/env python3
"""
Empathetic ChatBot Test Script
共感的なチャットボットのテストスクリプト
"""

from empathetic_chat import EmpatheticChatBot

def main():
    print("\n" + "="*60)
    print("🤖 Empathetic ChatBot - 共感的なAIチャットボット")
    print("="*60)
    
    # モデルの初期化
    print("\n📚 Loading model... This may take a moment...")
    chatbot = EmpatheticChatBot("microsoft/Phi-3-mini-4k-instruct")
    
    print("\n✨ Ready! Type your message or 'quit' to exit.")
    print("-"*60)
    
    # サンプル会話
    sample_inputs = [
        "今日は本当に疲れました。仕事が大変で...",
        "最近、友達との関係がうまくいかなくて悩んでいます",
        "quit"
    ]
    
    print("\n📝 Running sample conversation:")
    print("-"*40)
    
    for user_input in sample_inputs:
        if user_input.lower() == 'quit':
            print("\n👋 Thank you for using Empathetic ChatBot!")
            break
            
        print(f"\n👤 User: {user_input}")
        
        try:
            response = chatbot.generate_response(
                user_input, 
                max_length=200, 
                temperature=0.7
            )
            
            # 長い応答は切り詰める
            if len(response) > 300:
                response = response[:297] + "..."
                
            print(f"🤖 Bot: {response}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-"*40)

if __name__ == "__main__":
    main()