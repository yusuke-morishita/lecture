# Using PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
# Using OpenCV
import cv2
# Using numpy
import numpy as np

# GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define a nuural network model (LeNet)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace = True),
            nn.Linear(120, 84),
            nn.ReLU(inplace = True),
            nn.Linear(84, 2)
        )

    def forward(self, x):
        return self.classifier(x)

# Create a neural network model
model = Net().to(device)

# Load a model
model.load_state_dict(torch.load('model_weights.pth'))
#model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))

# Initialize a face detector by OpenCV
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize a camera capture
cam = cv2.VideoCapture(0)

model.eval()
with torch.no_grad():
    while True:
        # Capture one frame from camera
        ret, frame = cam.read()
        img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector.detectMultiScale(img_g)
        for x, y, w, h in faces:
            # Show detection results
            #print(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Create a norm image
            img_norm = cv2.resize(img_g[y:y + h, x:x + w], dsize = (32, 32))

            # Convert into pytorch format
            inputs = torch.from_numpy(img_norm.reshape((1, 1, 32, 32)).astype(np.float32))
            inputs = inputs.to(device)

            # Inference
            outputs = model(inputs)

            # Retrieve an inference result
            pred = outputs.argmax(1)
            print('Softmax: {0:.3f}'.format(np.exp(outputs[0, 1].item()) /
                (np.exp(outputs[0, 0].item()) + np.exp(outputs[0, 1].item()))))
            print('Result: {} => {}'.format(pred[0].item(), 'smile' if pred[0] > 0.5 else 'neutral'))

            # Show detection results
            if pred[0] > 0.5:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show image
        cv2.imshow('image', frame)
        key = cv2.waitKey(10)
        if key == 27 or key == 'q': # ESC or q
            break

    cam.release()
    cv2.destroyAllWindows()

