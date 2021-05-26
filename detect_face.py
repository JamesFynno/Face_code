from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import os

def detectFace(img):
    '''
    Hàm phát hiện khuôn mặt từ img và trả về 1 torch.Size([3, 240, 240])

    Param: path
    Return: torch.Size([3, 240, 240])
    '''
    img1 = Image.open(img)
    detector = MTCNN(image_size=240, margin=0, min_face_size=20) 
    face, prob = detector(img1, return_prob = True)
    # plt.imshow(face.permute(1,2,0))
    return face, prob

# Các hàm hỗ trợ load ảnh và lưu data
def getIDList(path):
    '''
    Input: path Images folder
    Output: ID list

    Hàm trả về list các ID của User
    '''
    return os.listdir(path)


def collate_fn(x):
    return x[0]

def loadDatasets(path):
    '''
    Hàm load data từ folder và lưu dữ liệu ảnh và prob dưới file pt

    Param: path
    Return: file data.pt  
    '''
    resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embedding conversion
    dataset = datasets.ImageFolder(path)
    loader = DataLoader(dataset, collate_fn=collate_fn)
    index_to_class = {i:c for c,i in dataset.class_to_idx.items()}
    
    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embedding matrix after conversion from cropped faces to embedding matrix using resnet

    for img, idx in loader:
        face, prob = detectFace(img)
        if face is not None and prob>0.90: # if face detected and porbability > 90%
            emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
            embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
            name_list.append(index_to_class[idx]) # names are stored in a list
        return embedding_list, name_list
    data = [embedding_list,name_list]
    torch.save(data, 'data1.db') # saving data.pt file


