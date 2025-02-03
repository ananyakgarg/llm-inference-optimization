import time
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_baseline_inference(
    model_name="gpt2-medium",
    prompt="Hey, I'm Ananya...",
    max_length=150,  
    do_sample=True,
    temperature=0.7,  
    top_k=30,         
    top_p=0.85,     
    repetition_penalty=1.2, 
    device_type="auto"
):
    """
    Optimized inference function with better generation parameters
    """
    # device selection
    if device_type == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_type)
        
    print(f"Using device: {device}")

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6  # More accurate memory tracking

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if device.type == "mps":
        torch.mps.empty_cache()

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,  
            pad_token_id=tokenizer.eos_token_id  
        )
    end_time = time.time()

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    mem_after = process.memory_info().rss / 1e6
    mem_consumed_mb = mem_after - mem_before
    inference_time = end_time - start_time

    print("\n" + "=" * 50)
    print(f"Memory consumed: {mem_consumed_mb:.2f} MB")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Output length: {len(outputs[0])} tokens")
    print("=" * 50 + "\n")

    return generated_text, inference_time

if __name__ == "__main__":
    text, time_taken = run_baseline_inference(
        model_name="gpt2-large",
        prompt="Hey, I'm Ananya. I love programming, AI, and building. You are a helpful assistant.",
        max_length=150,
        temperature=0.7,
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.2,
        device_type="auto"  
    )
    
    print("Generated Text:\n" + "-" * 50)
    print(text)
    print(f"\nTotal Inference Time: {time_taken:.2f} seconds")