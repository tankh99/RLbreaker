from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

# Load model and tokenizer
print("Loading RoBERTa Checkpoint...")
ckpt_path = 'hubert233/GPTFuzz'
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
model = RobertaForSequenceClassification.from_pretrained(ckpt_path).to('cuda')
tokenizer = RobertaTokenizer.from_pretrained(ckpt_path)
print("Loading Done!")

def predict(sequences):

    # Encoding sequences
    inputs = tokenizer(sequences, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')

    # Compute token embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predictions
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # If you want the most likely classes:
    _, predicted_classes = torch.max(predictions, dim=1)


    return predicted_classes

