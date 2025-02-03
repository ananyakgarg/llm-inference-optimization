import torch, os
from transformers import AutoModelForCausalLM

def apply_dynamic_quantization(model="gpt2-large"):
    model = AutoModelForCausalLM.from_pretrained(model)
    model.eval()  # eval mode

    model.cpu()

    if 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'
    else:
        raise RuntimeError("No qnnpack support")
    
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    return quantized_model


if __name__ == "__main__":
    quantized_model = apply_dynamic_quantization()
    print(quantized_model)
    
    save_dir = "../models/quantized_gpt2"
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(quantized_model.state_dict(), os.path.join(save_dir, "state_dict.pth"))