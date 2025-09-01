from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    print("Loading GPT-2 model...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "The future of artificial intelligence is"
    
    print(f"Prompt: {prompt}")
    print("Generating response...")
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{response}")

if __name__ == "__main__":
    main()