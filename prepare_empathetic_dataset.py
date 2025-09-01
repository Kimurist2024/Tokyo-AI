import json
from datasets import load_dataset, Dataset
from typing import List, Dict
import random

def prepare_empathetic_dialogues():
    """
    Empathetic Dialoguesデータセットを読み込み、
    Llama 3のファインチューニング用フォーマットに変換
    """
    print("Loading Empathetic Dialogues dataset...")
    try:
        # 新しい方法でデータセットを読み込み
        dataset = load_dataset("facebook/empathetic_dialogues", trust_remote_code=True)
    except RuntimeError:
        # 代替方法: parquetファイルから直接読み込み
        print("Using alternative loading method...")
        dataset = {
            'train': load_dataset("parquet", 
                                data_files="https://huggingface.co/datasets/facebook/empathetic_dialogues/resolve/main/train.parquet")['train'],
            'validation': load_dataset("parquet", 
                                     data_files="https://huggingface.co/datasets/facebook/empathetic_dialogues/resolve/main/validation.parquet")['train']
        }
    
    train_data = []
    val_data = []
    
    def format_conversation(example):
        """対話を指示-応答フォーマットに変換"""
        formatted_examples = []
        
        utterances = example['utterances']
        context = example['context']
        
        conversation = []
        for i, utterance in enumerate(utterances):
            conversation.append(utterance)
            
            if i > 0 and i % 2 == 1:
                instruction = f"Context: {context}\nUser: {conversation[-2]}"
                response = conversation[-1]
                
                formatted_examples.append({
                    "instruction": "Respond empathetically to the following message.",
                    "input": instruction,
                    "output": response
                })
        
        return formatted_examples
    
    print("Processing training data...")
    for example in dataset['train']:
        formatted = format_conversation(example)
        train_data.extend(formatted)
    
    print("Processing validation data...")
    for example in dataset['validation']:
        formatted = format_conversation(example)
        val_data.extend(formatted)
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    train_data = train_data[:10000]
    val_data = val_data[:1000]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    with open('empathetic_train.json', 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open('empathetic_val.json', 'w') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print("\nSample training example:")
    print(json.dumps(train_data[0], indent=2, ensure_ascii=False))
    
    return train_data, val_data

def create_prompt_template(example):
    """Llama 3用のプロンプトテンプレート"""
    system_prompt = "You are an empathetic assistant who provides supportive and understanding responses."
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['output']}<|eot_id|>"""
    
    return prompt

if __name__ == "__main__":
    train_data, val_data = prepare_empathetic_dialogues()
    
    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print("Files created:")
    print("- empathetic_train.json")
    print("- empathetic_val.json")