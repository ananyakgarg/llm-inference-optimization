from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


if torch.backends.mps.is_available():
    print("MPS backend is available.")
else:
    print("MPS backend is NOT available. Will use CPU.")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)


