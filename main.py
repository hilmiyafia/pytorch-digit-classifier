import torchvision.transforms as transforms
import subprocess
import torchinfo
import torch.onnx
import torch
import numpy
import tqdm
import cv2
import os

""" CREATE MODEl """
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 3, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(3, 6, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(6, 10, 2), torch.nn.Flatten()).cuda()
torchinfo.summary(model, (1, 16, 16))

""" LOAD DATA """
data = []
for i in range(10):
    image = cv2.imread(f"images/{i}.png") / 255
    data.append([numpy.max(image, -1)])
data = torch.FloatTensor(numpy.stack(data)).cuda()

""" CREATE OPTIMIZER """
optimizer = torch.optim.Adagrad(model.parameters())

""" CREATE LOSS FUNCTION """
loss = torch.nn.CrossEntropyLoss()

""" CREATE LABELS """
target = torch.arange(10, dtype=torch.long).cuda()

""" CREATE AUGMENTATION FUNCTION """
augment = torch.nn.Sequential(
    transforms.RandomPerspective(p=0.2),
    transforms.Resize((16, 16))).cuda()

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
dummy = torch.randn(1, 1, 16, 16)
torch.onnx.export(model, dummy, "m.onnx", verbose=True)
subprocess.call(["onnxsim", "m.onnx", "model.onnx"])
os.remove("m.onnx")
