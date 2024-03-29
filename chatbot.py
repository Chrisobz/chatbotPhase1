import random  
import json  

import torch  
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available, else CPU

# Load intents data from JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"  # Define file path
data = torch.load(FILE)  

input_size = data["input_size"] 
hidden_size = data["hidden_size"]
output_size = data["output_size"]  
all_words = data['all_words']  
tags = data['tags'] 
model_state = data["model_state"] 

model = NeuralNet(input_size, hidden_size, output_size).to(device)  # Initialize neural network model and move to device
model.load_state_dict(model_state)  
model.eval()  

bot_name = "Christian"  # 
print("Let's chat! (type 'quit' to exit)")  
while True:
    sentence = input("You: ")  
    if sentence == "quit": 
        break

    sentence = tokenize(sentence)  # Tokenize user input
    X = bag_of_words(sentence, all_words)  
    X = X.reshape(1, X.shape[0])  
    X = torch.from_numpy(X).to(device) 

    output = model(X)  # Get model output
    _, predicted = torch.max(output, dim=1)  # Get predicted tag

    tag = tags[predicted.item()]  # Get corresponding tag

    probs = torch.softmax(output, dim=1)  # Get probabilities
    prob = probs[0][predicted.item()]  
    if prob.item() > 0.75:  
        for intent in intents['intents']: 
            if tag == intent["tag"]:  
                print(f"{bot_name}: {random.choice(intent['responses'])}")  # Print bot response
    else:  
        print(f"{bot_name}: I do not understand...") 
