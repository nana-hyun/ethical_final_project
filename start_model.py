import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tkinter as tk
from tkinter import filedialog
import os

# Function to load txt file through file dialog
def load_txt_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select a TXT File",
        filetypes=[("Text files", "*.txt")]
    )
    return file_path

# Function to run inference on the loaded model
def run_inference(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions

if __name__ == "__main__":
    # Step 1: Load the saved model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = "./saved_model"  # Replace with your saved model path
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    labels = ["principle-oriented", "result-oriented", "social-oriented"]
    # Step 2: Select input TXT file
    print("Please select a TXT file containing input data.")
    txt_file = load_txt_file()

    if txt_file and os.path.exists(txt_file):
        # Step 3: Read content from the file
        with open(txt_file, "r", encoding="utf-8") as file:
            input_text = file.read().strip()
        
        # Step 4: Run inference and get predictions
        predictions = run_inference(model, tokenizer, input_text)
        
        # Step 5: Print results
        print("\nModel Predictions:")
        for i, score in enumerate(predictions[0]):
            print(f"{labels[i]}: {score.item():.4f}")
        print(f"The model predicts that the input text is {labels[predictions[0].argmax().item()]} oriented.")
    else:
        print("No file selected or file not found.")
