import torch
from transformers import BertTokenizerFast, BertModel
import torch.nn as nn
import sys

# defining the BERTClassifier class (matches training code)
class BERTClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.pooler_output
        return self.classifier(cls_output)

# loading the tokenizer and pre-trained BERT model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# creating an instance of BERTClassifier
model = BERTClassifier(bert_model)

# loading the saved checkpoint from training
model.load_state_dict(torch.load("best_bert_model.pt"))

# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  # Set to evaluation mode for prediction

# defining the predict function
def predict(title):
    """
    Predict whether an article title is fake news or not.
    
    Args:
        title (str): The article title to classify.
    
    Returns:
        dict: A dictionary with the predicted label and probability of fake news.
              - 'predicted_label': int (0 for real, 1 for fake)
              - 'probability_fake': float (probability of being fake)
    """
    # tokenize the input title (match training settings)
    inputs = tokenizer(
        title,
        return_tensors="pt",  # returning PyTorch tensors
        padding=True,
        truncation=True,
        max_length=15  # matching the max_title_length from training
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # pprediction without gradient computation
    with torch.no_grad():
        log_probs = model(input_ids, attention_mask)  # getting log probabilities
        probs = torch.exp(log_probs)  # convert to probabilities
        predicted_label = torch.argmax(probs, dim=1).item()  # predicted class (0 or 1)
        prob_fake = probs[0, 1].item()  # probability of class 1 (fake)

    return {
        'predicted_label': predicted_label,
        'probability_fake': prob_fake
    }

if __name__ == "__main__":
    print("Starting script...")
    if len(sys.argv) != 2:
        print("Usage: python3 bert_predict.py \"<article_title>\"")
        sys.exit(1)

    title = sys.argv[1]
    try:
        result = predict(title)
        print(f"Predicted Label: {result['predicted_label']} (0=real, 1=fake)")
        print(f"Probability of Fake: {result['probability_fake']:.4f}")
    except Exception as e:
        print(f"Error: {e}")