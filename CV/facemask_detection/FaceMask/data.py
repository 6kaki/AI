import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.nn.init
from utils import utils
import cv2
from PIL import Image
import numpy as np



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,size=0,transform=None):
        super(CustomDataset,self).__init__()

        self.root = utils.archive_path
        self.size = size
        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        if self.size == 0:
            self.label_path_list = utils.search_file('annotations')
            self.img_path_list = utils.search_file('images')

        else:
            self.label_paths = utils.reshape_annotations(size, 'annotations')
            self.img_paths = utils.reshape_img(size, 'images')

            self.label_path_list = utils.search_file(self.label_paths.split('\\')[-1])
            self.img_path_list = utils.search_file(self.img_paths.split('\\')[-1])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        if self.size == 0:
            img = cv2.imread(self.img_path_list[item])
            img = self.transforms(img).to(self.device)
            label_name = self.label_path_list[item]
            _, label = utils.read_xml(label_name)
            targets = utils.make_target(label)
            complete_target = []
            for t in targets:
                complete_target.append(t)
        else:
            img = cv2.imread(self.img_path_list[item])
            img = self.transforms(img).to(self.device)
            label_name = self.label_path_list[item]
            label = utils.read_resized_annotation(self.size,label_name)
            targets = utils.make_target(label)
            complete_target = []
            for t in targets:
                t['boxes'] = torch.FloatTensor(t['boxes'])
                t['labels'] = torch.IntTensor(t['labels'])

                t['boxes'] = t['boxes'].to(self.device)
                t['labels'] = torch.as_tensor(t['labels'],dtype=torch.int64)
                complete_target.append(t)

        return img, complete_target


def get_transform():
    transforms = []
    transforms.append(torchvision.transforms.PILToTensor())
    transforms.append(torchvision.transforms.ConvertImageDtype(torch.float),)
    return torchvision.transforms.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    # test = CustomDataset(size=144)
    #
    # fig = test[10]
    # print(fig['image'].shape, fig['label'])
    #
    # utils.img_with_bnd(test.img_path_list[10], fig['label'])





    from torch.utils.data import random_split
    from tqdm.notebook import tqdm
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    import numpy as np

    transformed_dataset = CustomDataset(
        size=0,
        transform=get_transform())

    dataset_size = len(transformed_dataset)
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - val_size

    train_dataset,val_dataset,test_dataset = random_split(
        transformed_dataset
        ,[train_size,val_size,test_size]
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )



    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 4
    num_epochs = 10
    batch_size = 4

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretranined=True)
    # 분류기 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for imgs,y in tqdm(dataloader):

            imgs = list(img.to(device) for img in imgs)
            print(y)
            print(y[0])
            print(y[0][0])
            annotations = [{k:v.to(device) for k,v in t.items()} for t in y]
        #
        #     predict = model(imgs,y)
        #     losses = sum(loss for loss in predict.values())
        #
        #     optimizer.zero_grad()
        #     losses.backward()
        #     optimizer.step()
        #     epoch_loss += losses
        #
        # print(f'{epoch+1}/{num_epochs} {epoch_loss}')
            break
        break




