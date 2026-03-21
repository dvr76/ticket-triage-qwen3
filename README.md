# Fine-Tuning Qwen3 for Structured Property Maintenance Ticket Triaging

Fine-tuned Qwen3-0.6B for extracting structured maintenance information from tenant ticket text.

## Training

- **Base model:** Qwen/Qwen3-0.6B (loaded via unsloth/Qwen3-0.6B)
- **Method:** QLoRA
- **Framework:** Unsloth + TRL SFTTrainer
- **Hardware:** Google Colab T4 (16GB VRAM)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "dvr76/ticket-triage-qwen3"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are a property maintenance ticket triage system. Respond with ONLY valid JSON."},
    {"role": "user", "content": "kitchen sink tap water is leaking from yesterday morning"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Output schema

```json
{
  "is_maintenance_request": true,
  "issues": [
    { "category": "", "sub_category": "", "location": "", "urgency": "" }
  ],
  "vendor_type": "",
  "entry_required": true
}
```
