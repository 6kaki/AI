import os

import torchvision.models.detection
import torchvision.transforms
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
from bs4 import BeautifulSoup
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
from tqdm.notebook import tqdm
from data import CustomDataset
from utils import utils


## 해당 폴더에 들어있는 파일 이름들을 가져온다.
## 'maksssksksss0.png'...
imgs = list(sorted(os.listdir('../archive/images')))
print(imgs)

## 파일 이름을 기준으로
## train과 test set으로 나눈다.
train_imgs, test_imgs = train_test_split(imgs)
print(len(train_imgs), len(test_imgs))

def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)

    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    return [xmin,ymin,xmax,ymax]

def generate_label(obj):
    if obj.find('name').text == 'with_mask':
        return 1
    elif obj.find('name').text == 'mask_weared_incorrect':
        return 2
    elif obj.find('name').text == 'without_mask':
        return 3
    return 0

class MaskDataset(torch.utils.data.Dataset):
    def __init__(self,img_dir,ann_dir,image_list,transforms):
        self.transforms = transforms
        self.imgs = image_list
        self.img_dir, self.ann_dir = img_dir, ann_dir

    def __getitem__(self, idx):
        file_image = 'maksssksksss' + str(idx) + '.png'
        file_label = 'maksssksksss' + str(idx) + '.xml'

        img_path = os.path.join(self.img_dir, file_image)
        label_path = os.path.join(self.ann_dir, file_label)

        img = Image.open(img_path).convert('RGB')
        target = self.__generate_target(idx,label_path)

        if self.transforms is not None:
            img,target = self.transforms(img,target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    ## 객체를 만들지 않고도 함수를 사용할 수 있는 기능
    @staticmethod
    def __generate_target(image_id, file):
        with open(file) as f:
            data = f.read()
            soup = BeautifulSoup(data,'xml')
            objects = soup.find_all('object')

            num_objs = len(objects)

            boxes = []
            labels = []
            for i in objects:
                boxes.append(generate_box(i))
                labels.append(generate_label(i))

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            img_id = torch.tensor([image_id])
            area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
            iscrowd = torch.zeros((num_objs),dtype=torch.int64)

            target = {}
            target['boxes'] = boxes
            target['labels'] = labels
            target['image_id'] = img_id
            target['area'] = area
            target['iscrowd'] = iscrowd
            return target

def get_model_instance_segmentation(num_classes, pretranined=True):
    # pretrained model 불러오기
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretranined=pretranined)

    #분류기 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
    return model

def get_transform():
    transforms = []
    ## img -> tensor로 변환
    transforms.append(torchvision.transforms.PILToTensor())
    ## 0~1로 변환
    transforms.append(torchvision.transforms.ConvertImageDtype(torch.float),)
    return torchvision.transforms.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def main(num_epochs = 10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = MaskDataset(
        '../archive/images',
        '../archive/annotations',
        image_list = train_imgs,
        transforms=get_transform()
    )

    dataset_test = MaskDataset(
        '../archive/images',
        '../archive/annotations',
        image_list = test_imgs,
        transforms=get_transform()
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        bacth_size = 1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    num_classes = 4
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,lr=0.005,momentum=0.9,weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs)

    for epoch in range(num_epochs):

        for X,y in tqdm(data_loader):
            X = list(img.to(device) for img in X)
            y = [{k:v.to(device) for k,v in t.items()} for t in y]
            predict = model(X,y)
            h = model(X)

            lr_scheduler.step()


        torch.save(model.state_dict(),'checkpoint.pth')


main(num_epochs=8)