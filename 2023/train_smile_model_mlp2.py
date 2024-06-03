# Using PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# ===== Training params =====
# training parameters
num_epochs = 10
num_batch = 64
learning_rate = 0.001

# GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== Training Data  =====
# Load a training dataset
dataset = torch.load('smile_dataset.pt')
#print(len(dataset))

# Split the dataset: training (80%) / testing (20%)
n_dataset = len(dataset)
n_train = int(n_dataset * 0.8)
n_test = n_dataset - n_train
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])

# Set a dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = num_batch, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = num_batch, shuffle = False)

# ===== Training model =====
# Define a nuural network model (MLP)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(x)

# Create a neural network model
model = Net().to(device)
# Set a loss function
criterion = nn.CrossEntropyLoss()
# Set an optimization method
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# ==== Training process ====
# Set training mode
model.train()

# Run training
for epoch in range(num_epochs):
    loss_sum = 0
    for inputs, labels in train_dataloader:
        # Flatten
        inputs = inputs.view(-1, 1*32*32)
        # Transfer data to GPU if needed
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Initialize the optimizer
        optimizer.zero_grad()
        # Process the NN model
        outputs = model(inputs)
        # Compute the loss value
        loss = criterion(outputs, labels)
        loss_sum += loss
        # Compute the gradient
        loss.backward()
        # Update weights
        optimizer.step()

    print(f"[Train] Epoch {epoch + 1}/{num_epochs}: Loss = {loss_sum.item() / len(train_dataloader)}")

    # Save the trained model
    torch.save(model.state_dict(), 'model_weights.pth')

# ====== Test process ======
# Set evaluation mode
model.eval()
loss_sum = 0
correct = 0

# Run evaluation
with torch.no_grad():
    for inputs, labels in test_dataloader:
        # Flatten
        inputs = inputs.view(-1, 1*32*32)
        # Transfer data to GPU if needed
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Process the NN model
        outputs = model(inputs)
        # Compute the loss value
        loss_sum += criterion(outputs, labels)
        # Retrieve the prediction value (1=smile or 0=neutral)
        prediction = outputs.argmax(1)
        # Count the number of correct predictions
        correct += prediction.eq(labels.view_as(prediction)).sum().item()

    print(f"[Test] Loss = {loss_sum.item() / len(test_dataloader)}, Accuracy: {100 * correct / len(test_dataset)}%")

