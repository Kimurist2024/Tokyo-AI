import json
import requests
from typing import List, Dict
import random

def create_sample_empathetic_data():
    """
    Empathetic Dialoguesが利用できない場合のサンプルデータ作成
    実際の共感的対話例を手動で作成
    """
    
    sample_data = [
        {
            "instruction": "Respond empathetically to the following message.",
            "input": "I'm feeling really overwhelmed with work lately. I have so many deadlines and I don't know how to manage everything.",
            "output": "That sounds incredibly stressful, and I can understand why you're feeling overwhelmed. Having multiple deadlines can feel like a lot of pressure. It's completely normal to feel this way when you have so much on your plate. Would you like to talk about ways to prioritize your tasks, or would you prefer to just share what's been weighing on you most?"
        },
        {
            "instruction": "Respond empathetically to the following message.",
            "input": "I just found out my best friend has been talking behind my back. I feel so betrayed and hurt.",
            "output": "I'm so sorry you're going through this. Feeling betrayed by someone close to you is one of the most painful experiences, and your feelings are completely valid. It must be especially hard since this is your best friend - someone you trusted deeply. Take your time processing these emotions. Have you been able to talk to anyone else about how you're feeling?"
        },
        {
            "instruction": "Respond empathetically to the following message.",
            "input": "I failed my driving test again today. This was my third attempt and I'm starting to think I'll never get my license.",
            "output": "Oh no, I can imagine how disappointing and frustrating that must be, especially after putting in the effort for your third attempt. It's completely understandable to feel discouraged right now. Learning to drive can be challenging, and everyone progresses at their own pace. The fact that you've tried three times shows your determination and persistence. Many people need multiple attempts - you're definitely not alone in this."
        },
        {
            "instruction": "Respond empathetically to the following message.",
            "input": "My grandmother passed away last week. I'm having a hard time dealing with the grief.",
            "output": "I'm so deeply sorry for your loss. Losing a grandmother is incredibly painful, and grief can feel overwhelming. Please know that what you're feeling is completely natural and there's no 'right' way to grieve. Your grandmother clearly meant a lot to you, and it's beautiful that you had such a meaningful relationship with her. Take all the time you need to process your emotions. Is there anything specific about her that you'd like to share or remember right now?"
        },
        {
            "instruction": "Respond empathetically to the following message.",
            "input": "I got rejected from my dream college today. I've been working towards this for years and I feel like all my effort was wasted.",
            "output": "I'm so sorry to hear about the rejection. That must be absolutely heartbreaking, especially after years of dedication and hard work toward this goal. Your disappointment is completely understandable - when we invest so much of ourselves into something, a setback like this can feel devastating. But please know that your efforts weren't wasted. The skills, knowledge, and determination you've developed are yours to keep, and they will serve you well wherever your path leads next."
        },
        {
            "instruction": "Respond empathetically to the following message.",
            "input": "I'm going through a divorce and I feel like my whole world is falling apart.",
            "output": "I'm so sorry you're going through such a difficult time. Divorce can feel like everything you knew is changing, and that sense of your world falling apart is so real and valid. It's one of life's most challenging experiences, and it's completely normal to feel overwhelmed, scared, or uncertain about the future. Please be gentle with yourself during this time. You don't have to have everything figured out right now. Are you getting support from friends, family, or a counselor during this transition?"
        },
        {
            "instruction": "Respond empathetically to the following message.",
            "input": "I've been struggling with anxiety and depression lately. Some days I don't even want to get out of bed.",
            "output": "Thank you for sharing something so personal with me. What you're experiencing sounds really difficult, and I want you to know that your feelings are valid. Anxiety and depression can make even simple tasks feel overwhelming, and those days when getting out of bed feels impossible are more common than you might think. You're not alone in this struggle. It takes courage to acknowledge these feelings. Have you been able to reach out to a mental health professional, or do you have support from friends or family?"
        },
        {
            "instruction": "Respond empathetically to the following message.",
            "input": "I lost my job yesterday after 10 years with the company. I don't know what I'm going to do.",
            "output": "That's such devastating news, and I can only imagine the shock and uncertainty you must be feeling right now. Losing a job after 10 years isn't just about losing income - it's losing a huge part of your routine, identity, and sense of security. Your feelings of confusion and worry are completely natural. After dedicating a decade of your life to this company, this transition must feel overwhelming. Please take time to process this change. You have 10 years of valuable experience and skills that no one can take away from you."
        }
    ]
    
    # 各サンプルを複数のバリエーションで拡張
    extended_data = []
    
    # 元のデータを追加
    extended_data.extend(sample_data)
    
    # より多くのサンプルを生成（異なる表現でのバリエーション）
    additional_samples = [
        {
            "instruction": "Respond empathetically to the following message.",
            "input": "I'm having trouble sleeping because I keep worrying about everything.",
            "output": "Sleep troubles caused by worrying can be so exhausting - you're dealing with both the stress of your concerns and the fatigue from not sleeping well. It's like a difficult cycle. Your mind racing with worries at bedtime is something many people experience, especially during stressful periods. It sounds like you have a lot on your mind right now. Sometimes it can help to write down your worries before bed, or try some gentle breathing exercises. What kinds of things have been keeping you up at night?"
        },
        {
            "instruction": "Respond empathetically to the following message.", 
            "input": "I feel so lonely even when I'm surrounded by people.",
            "output": "That's such a profound and difficult feeling to experience. Feeling lonely in a crowd can sometimes feel even more isolating than being alone, because it highlights a disconnect between what others see and what you're experiencing inside. You're not alone in feeling this way - many people struggle with this kind of loneliness, where the connections around us don't feel deep or authentic enough. It takes courage to recognize and acknowledge these feelings. What do you think might help you feel more genuinely connected to others?"
        }
    ]
    
    extended_data.extend(additional_samples)
    
    return extended_data

def prepare_training_data():
    """訓練データの準備"""
    print("Creating sample empathetic dialogue data...")
    
    # サンプルデータ作成
    sample_data = create_sample_empathetic_data()
    
    # データを訓練用と検証用に分割
    random.shuffle(sample_data)
    split_point = int(len(sample_data) * 0.8)
    
    train_data = sample_data[:split_point]
    val_data = sample_data[split_point:]
    
    # 訓練データを複数回複製してデータ量を増やす
    train_data = train_data * 50  # 50倍に拡張
    val_data = val_data * 10     # 10倍に拡張
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # JSONファイルに保存
    with open('empathetic_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open('empathetic_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print("\nSample training example:")
    print(json.dumps(train_data[0], indent=2, ensure_ascii=False))
    
    return train_data, val_data

if __name__ == "__main__":
    train_data, val_data = prepare_training_data()
    
    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print("Files created:")
    print("- empathetic_train.json")
    print("- empathetic_val.json")
    print(f"Total training samples: {len(train_data)}")
    print(f"Total validation samples: {len(val_data)}")