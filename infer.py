import torch
import cv2
from torchvision import transforms
from model.model import model 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()

model = model()
model = model.to(DEVICE)
# image = Image.open("../dataset/test.jpg")
image = cv2.imread("dataset/test.jpg")

image = transform(image)
image = image.to(DEVICE)

image = image.unsqueeze(0)
image = image.unsqueeze(0)

print(image.shape)
features = model(image)
print(features.shape)

print(features)
