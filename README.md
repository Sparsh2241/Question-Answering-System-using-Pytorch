# Question Answering System

## Overview
This project implements a **Question Answering System** using **LSTM (Long Short-Term Memory)** with **PyTorch**. The system is trained on a question-answering dataset, where it undergoes tokenization, preprocessing, and conversion of categorical data into numerical format. The project includes:

- Tokenization and preprocessing of text data.
- Encoding categorical data into numerical format.
- Creating a **custom class** for dataset management.
- Implementing a **DataLoader** for efficient batch processing.
- Building and training an **LSTM Model** with:
  - **Learning rate:** 0.001
  - **Epochs:** 50
- Performing **predictions** after training.

---


---

## Dataset
The dataset consists of **question-answer pairs**. The preprocessing steps include:
1. Tokenizing the text data.
2. Cleaning and normalizing text.
3. Converting categorical data (words) into numerical representations.

---

## Implementation Details

### 1. Data Preprocessing
- Tokenization is performed using `nltk` or similar libraries.
- Text data is cleaned and normalized.
- Questions and answers are converted into numerical format using an encoding method.

### 2. Dataset Class & DataLoader
A custom PyTorch Dataset class is created to load and process the dataset:

```python
from torch.utils.data import Dataset, DataLoader

class QADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question_tokens = self.tokenizer(item['question'])
        answer_label = item['answer_label']
        return {
            'input': torch.tensor(question_tokens, dtype=torch.long),
            'label': torch.tensor(answer_label, dtype=torch.long)
        }

# Example usage:
# dataset = QADataset(data, tokenizer_function)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 3. LSTM Model
The LSTM model is implemented in PyTorch:

```python
import torch
import torch.nn as nn

class LSTMQA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMQA, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
```

### 4. Training the Model

```python
import torch.optim as optim

input_dim = 100  # Example input dimension (e.g., embedding size)
hidden_dim = 128
output_dim = 10  # Number of output classes
num_layers = 2

model = LSTMQA(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = batch['input']
        labels = batch['label']
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

### 5. Prediction
After training, predictions are made using:

```python
model.eval()
with torch.no_grad():
    test_question = "What is the capital of France?"
    test_tokens = tokenizer(test_question)
    test_input = torch.tensor(test_tokens, dtype=torch.long).unsqueeze(0)
    prediction = model(test_input)
    predicted_label = prediction.argmax(dim=1).item()
    print("Predicted Answer Label:", predicted_label)
```

---

## Conclusion
This project successfully implements a **Question Answering System** using **LSTM in PyTorch**. The system undergoes preprocessing, training, and evaluation, and is capable of predicting answers based on input questions. The model can be further improved by experimenting with hyperparameters, embeddings, and dataset augmentation.




