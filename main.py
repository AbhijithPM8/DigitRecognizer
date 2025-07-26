import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2
from model import Net

model=Net()
model.load_state_dict(torch.load('model/digit_model.pth',map_location=torch.device('cpu')))
model.eval()

transform=transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
])

canvas=np.zeros((400,400),dtype=np.uint8)
drawing=False
ix,iy=-1,-1


def draw(event,x,y,flags,param):
    global drawing,ix,iy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
    elif event==cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas,(ix,iy),(x,y),(255),thickness=20)
        ix,iy=x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing=False
        
cv2.namedWindow('Draw Digit')
cv2.setMouseCallback('Draw Digit',draw)

while True:
    display=canvas.copy()
    cv2.imshow('Draw Digit',display)
    key=cv2.waitKey(1)
    
    if key==ord('p'):
        img=cv2.resize(canvas,(28,28))
        img=Image.fromarray(img)
        img=transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output=model(img)
            prediction=torch.argmax(output,dim=1).item()
            print(f"Prediction : {prediction}")
    elif key==ord('c'):
        canvas[:]=0
    elif key=='27':
        break
    
cv2.destroyAllWindows()

