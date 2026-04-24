import torchvision.transforms as transforms
import torch.onnx
import torch
import numpy
import tqdm
import cv2
import os

""" CREATE MODEl """
model = torch.nn.Sequential()
model.append(torch.nn.Conv2d(1, 2, 3))
model.append(torch.nn.ReLU())
model.append(torch.nn.MaxPool2d(2))
model.append(torch.nn.Conv2d(2, 4, 3))
model.append(torch.nn.ReLU())
model.append(torch.nn.MaxPool2d(2))
model.append(torch.nn.Conv2d(4, 8, 3))
model.append(torch.nn.ReLU())
model.append(torch.nn.MaxPool2d(2))
model.append(torch.nn.Conv2d(8, 10, 2))
model.append(torch.nn.Flatten())

""" LOAD DATA """
data = []
for i in range(10):
    image = cv2.imread(f"images/{i}.png") / 255
    data.append([numpy.max(image, -1)])
data = numpy.array(data)
data = torch.tensor(data).type(torch.FloatTensor)

""" CREATE OPTIMIZER """
optimizer = torch.optim.Adagrad(model.parameters())

""" CREATE LOSS FUNCTION """
loss = torch.nn.CrossEntropyLoss()

""" CREATE LABELS """
target = torch.arange(10, dtype=torch.long)

""" CREATE AUGMENTATION FUNCTION """
augment = torch.nn.Sequential()
augment.append(transforms.RandomPerspective())
augment.append(transforms.Resize((32, 32)))

""" TRAINING """
bar = tqdm.tqdm(range(1000))
for j in bar:
    optimizer.zero_grad()
    output = model(augment(data))
    error = loss(output, target)
    error.backward()
    bar.set_postfix(loss=f"{error.item():.4f}")
    optimizer.step()

""" RESULTS """
print("Confusion Matrix")
matrix = numpy.zeros((10, 10))
for j in range(1000):
    output = model(augment(data))
    index = output.argmax(1)
    for l in range(10):
        matrix[l, index[l].item()] += 1

print(matrix)
accuracy = numpy.sum(numpy.eye(10) * matrix) / numpy.sum(matrix)
print(f"Accuracy: {accuracy * 100:.2f}%")

""" EXPORT MODEL """
dummy = torch.randn(1, 1, 32, 32)
torch.onnx.export(model, dummy, "model.onnx", verbose=True)
