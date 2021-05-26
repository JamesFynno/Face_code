from detect_face import *
def face_match(img_path, data_path):
    #gọi hàm xác định khuôn mặt
    face, prob = detectFace(img_path)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    if face is not None and prob>0.90:
    #trả về một ảnh cắt khuôn mặt và tỉ lệ % độ chính xác
        emb = resnet(face.unsqueeze(0)).detach()
        #gradient false
        saved_data = torch.load('data.pt')
        embedding_list = saved_data[0]
        name_list = saved_data[1]
        dist_list = [] #danh sách khoảng cách của test và vector train

    for idx, emb_db in enumerate(embedding_list):
        #tính khoảng cách giữa emb test với emb database trong file pt
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)
    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list))

result = face_match("Data/val/1/40758252_946160555572131_3730484984146821120_n.jpg", "data.pt")

print('Face matched with: ', result[0], 'with distance: ', result[1])
