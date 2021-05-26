from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

#khởi tạo MTCNN để phát hiện khuôn mặt
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
#khởi tạo resnet để nhúng khuôn mặt thành vector
resnet = InceptionResnetV1(pretrained='vggface2').eval()

#tạo dataset lưu trữ những bức hình cần train
dataset = datasets.ImageFolder('Data/train/')
#truy cập tên của mọi người trong folder dữ liệu
index_to_class = {i:c for c,i in dataset.class_to_idx.items()}
#khởi tạo hàm đối chiếu 
def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)
#print(type(loader))
face_list = [] #danh sách khuôn mặt đã cắt trong folder
name_list = [] #danh sách tên của những khuôn mặt
embedding_list = [] #danh sách các ma trận khuôn mặt đã được nhúng ở bước Resnet

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True)
    print(prob)
    #nếu khuôn mặt được xác định 
    #và đánh giá tỉ lệ % độ chính xác mà module detect được
    if face is not None and prob>0.95:
        #chuyển khuôn mặt đã được cắt sang resnet model để nhúng thành vector
        emb = resnet(face.unsqueeze(0))
        #chèn kết quả vào danh sách embedding_list
        embedding_list.append(emb.detach())
        #tên của người đó cũng được thêm vào danh sách 
        name_list.append(index_to_class[idx])

data = [embedding_list, name_list]
torch.save(data, "data.pt")

