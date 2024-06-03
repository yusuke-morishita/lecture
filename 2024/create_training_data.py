# Using OpenCV
import cv2
# Using numpy
import numpy as np
# Using PyTorch
import torch
import torch.utils.data

# Initialize a face detector by OpenCV
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read a file list
with open('lfw_with_smile_label.txt') as f:
    file_list = [s.strip().split() for s in f.readlines()]

    # Initialize a training data
    data_x = []
    data_y = []

    for filename, smile_label in file_list:
        # Read one image from file list
        print(filename, smile_label)
        img_bgr = cv2.imread(filename)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector.detectMultiScale(img_gray)

        # Show detection results
        #for x, y, w, h in faces:
        #    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #cv2.imshow('', img_bgr)
        #cv2.waitKey(0)

        # Create one norm image
        for x, y, w, h in faces:
            img_norm = cv2.resize(img_gray[y:y + h, x:x + w], dsize = (32, 32))
            data_x.append(img_norm)
            data_y.append(int(smile_label))
            #cv2.imshow('', img_norm)
            #cv2.waitKey(0)
            break

    # Create a training data
    data_x = np.expand_dims(np.array(data_x), axis = 1)  # (N,H=32,W=32)->(N,C=1,H=32,W=32)
    data_y = np.array(data_y)                            # (N,1)
    data_x = torch.tensor(data_x, dtype = torch.float32)
    data_y = torch.tensor(data_y, dtype = torch.int64)
    dataset = torch.utils.data.TensorDataset(data_x, data_y)
    # Save the training data
    torch.save(dataset, 'smile_dataset.pt')

