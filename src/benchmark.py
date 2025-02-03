import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def benchmark_inference(model, tokenizer, prompt, n_runs=20, max_length=150, **generate_kwargs):
    """
    Benchmarks the inference speed of a given model by running multiple generation runs.
    A warm-up run is executed first to eliminate one-time overhead.
    
    Returns:
        avg_time (float): Average inference time over n_runs.
        std_time (float): Standard deviation of inference time.
        times (list): List of inference times for each run.
    """
    # prepare input 
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    
    print("[INFO] Starting warm-up run...")
    # warm-up run
    with torch.no_grad():
        _ = model.generate(**inputs, max_length=max_length, **generate_kwargs)
    print("[INFO] Warm-up run complete.\n")
    
    # run multiple inference runs and time them
    times = []
    for i in range(n_runs):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, **generate_kwargs)
        elapsed = time.time() - start
        # tensor of shape [batch_size, sequence_length]
        token_count = outputs.shape[1] if outputs.dim() > 1 else len(outputs)
        times.append(elapsed)
        print(f"[Run {i+1}/{n_runs}] Completed in {elapsed:.4f} seconds; Generated {token_count} tokens.")
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    return avg_time, std_time, times

def load_baseline_model(model_name="gpt2-large", device=torch.device("cpu")):
    """
    Loads the baseline GPT-2 model.
    """
    print("[INFO] Loading baseline model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model

def apply_dynamic_quantization(model_name="gpt2-large"):
    """
    Loads the model and applies dynamic quantization (using qnnpack) to it.
    """
    print("[INFO] Loading model for dynamic quantization...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.cpu()
    if 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'
    else:
        raise RuntimeError("qnnpack support is required but not available.")
    
    print("[INFO] Applying dynamic quantization...")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model

def load_qat_model(model_name="gpt2-large", state_dict_path=os.path.join("models", "quantized_gpt2_qat_state_dict.pth")):
    """
    Loads a GPT-2 model and attempts to load the QAT quantized state_dict.
    """
    print("[INFO] Loading model for QAT quantized inference...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.cpu()
    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("[INFO] QAT quantized model loaded from state_dict.")
    else:
        print(f"[WARN] QAT quantized model state_dict not found at '{state_dict_path}'.")
    return model

def main():
    device = torch.device("cpu")
    model_name = "gpt2-large"
    prompt = "Hey, I'm Ananya. I love programming, AI, and building. You are a helpful assistant."
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_kwargs = {
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 30,
        "top_p": 0.85,
        "repetition_penalty": 1.2,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # --- Benchmark Baseline Model ---
    baseline_model = load_baseline_model(model_name=model_name, device=device)
    print("\n[Benchmark] Baseline Model")
    baseline_avg, baseline_std, _ = benchmark_inference(
        baseline_model, tokenizer, prompt, n_runs=20, max_length=150, **generate_kwargs
    )
    print(f"\n[RESULT] Baseline: Average Inference Time = {baseline_avg:.4f} sec (Std: {baseline_std:.4f} sec)")
    
    # --- Benchmark Dynamic Quantization Model ---
    dynamic_model = apply_dynamic_quantization(model_name=model_name)
    print("\n[Benchmark] Dynamic Quantization Model")
    dynamic_avg, dynamic_std, _ = benchmark_inference(
        dynamic_model, tokenizer, prompt, n_runs=20, max_length=150, **generate_kwargs
    )
    print(f"\n[RESULT] Dynamic Quantization: Average Inference Time = {dynamic_avg:.4f} sec (Std: {dynamic_std:.4f} sec)")
    
    # --- Benchmark QAT Quantized Model ---
    qat_model = load_qat_model(model_name=model_name)
    print("\n[Benchmark] QAT Quantized Model")
    qat_avg, qat_std, _ = benchmark_inference(
        qat_model, tokenizer, prompt, n_runs=20, max_length=150, **generate_kwargs
    )
    print(f"\n[RESULT] QAT Quantized: Average Inference Time = {qat_avg:.4f} sec (Std: {qat_std:.4f} sec)")
    
    # --- Summary ---
    print("\n" + "=" * 50)
    print("Benchmark Summary:")
    print(f"Baseline Model:       {baseline_avg:.4f} sec (Std: {baseline_std:.4f} sec)")
    print(f"Dynamic Quantization: {dynamic_avg:.4f} sec (Std: {dynamic_std:.4f} sec)")
    print(f"QAT Quantized:        {qat_avg:.4f} sec (Std: {qat_std:.4f} sec)")
    print("=" * 50)

if __name__ == "__main__":
    main()
