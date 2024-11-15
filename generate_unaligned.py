import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

questions = pd.read_csv('./datasets/eval/RL_Meta-Llama-3-8B-Instruct_0_responses_none.csv')['question'].tolist()
device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
model_name = "cognitivecomputations/Wizard-Vicuna-7B-Uncensored"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Handles multi-GPU or CPU deployment
    trust_remote_code=True,  # Required for some models
    torch_dtype=torch.float16,  # Use FP16 precision
    revision="main",  # Specify model revision/branch
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=1024, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

responses = []

for q in questions:
    response = generate_response(q)
    print(f"Generating response for: {q}")
    print(f"Response: {response}")
    print("-------------------")
    responses.append(response)


df = pd.DataFrame({'question': questions, 'response': responses})
df.to_csv('./datasets/question_unalign_res.csv', index=False)