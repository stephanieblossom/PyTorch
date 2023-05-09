import torch
import torchtext
from torchtext.data.utils import get_tokenizer

# Load pre-trained sentiment analysis model
model = torchtext.legacy.models.TextClassificationModel.load('sentiment_model.pt')

# Set up tokenizer
tokenizer = get_tokenizer('basic_english')

# Define sentiment labels
sentiment_labels = ['Negative', 'Positive']

# Function to predict sentiment
def predict_sentiment(text):
    # Tokenize input text
    tokens = tokenizer(text)

    # Convert tokens to numerical representation
    numerical_tokens = [model.vocab.stoi[token] for token in tokens]

    # Convert to PyTorch tensor
    tensor = torch.LongTensor(numerical_tokens).unsqueeze(1)

    # Perform prediction
    prediction = model(tensor)

    # Get predicted label
    predicted_label = sentiment_labels[prediction.argmax().item()]

    return predicted_label

# Get user input
user_input = input("Enter a text: ")

# Predict sentiment
sentiment = predict_sentiment(user_input)

# Print the predicted sentiment
print("Predicted sentiment:", sentiment)
