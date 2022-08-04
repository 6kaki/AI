import numpy as np
from selectivesearch import selective_search
import cv2
import utils
import data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms



def show_region_proposal(img,regions,limit):
    cnt = 0
    for r in regions:
        if r['size'] >= limit:
            cnt+=1
            rect = r['rect']
            left = rect[0]
            top = rect[1]
            right = left + rect[2]
            bottom = top + rect[3]
            cv2.rectangle(img,(left,top),(right,bottom),(0,256,0),0)
    print(f'Num of bbox : {cnt}')
    cv2.imshow('img',img)
    cv2.waitKey(0)


def compute_iou(candidate_box, bnd_box):

    x1 = np.maximum(candidate_box[0], bnd_box[0])
    y1 = np.maximum(candidate_box[1], bnd_box[1])
    x2 = np.minimum(candidate_box[2], bnd_box[2])
    y2 = np.minimum(candidate_box[3], bnd_box[3])

    intersection = np.maximum(x2 - x1,0) * np.maximum(y2 - y1,0)
    candi_area = (candidate_box[2] - candidate_box[0]) * (candidate_box[3] - candidate_box[1])
    bnd_box_area = (bnd_box[2]- bnd_box[0]) * (bnd_box[3] - bnd_box[1])
    union = candi_area + bnd_box_area - intersection
    iou = intersection / union
    return iou


def show_iou(bnd_box, regions, iou_over, iou_under):
    candi_region = [cand['rect'] for cand in regions if cand['size']]
    positive_sample = []
    negative_sample = []
    for bnd in bnd_box:
        if bnd[0] == 0:
            cv2.rectangle(img, (bnd[1], bnd[2]), (bnd[3], bnd[4]), (0, 0, 255), 0)
        else:
            cv2.rectangle(img, (bnd[1], bnd[2]), (bnd[3], bnd[4]), (0, 255, 0), 0)

        for idx, candi in enumerate(candi_region):
            candi_box = list(candi)
            candi_box[2] += candi_box[0]
            candi_box[3] += candi_box[1]

            iou = compute_iou(candi_box, bnd[1:])

            if iou > iou_over:
                print(f'index : {idx}  iou : {iou}')
                cv2.rectangle(img, (candi_box[0], candi_box[1]), (candi_box[2], candi_box[3]), (255, 0, 0), 0)
                text = "{}:{:.2f}".format(idx, iou)
                cv2.putText(img, text, (candi_box[0] + 1, candi_box[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            color=(255, 0, 0), thickness=1)
                positive_sample.append({'index':idx,'iou':iou,'rect':candi_box})
            elif iou < iou_over:
                negative_sample.append({'index':idx,'iou':iou,'rect':candi_box})
    cv2.imshow('iou_img', img)
    cv2.waitKey(0)
    return positive_sample, negative_sample






if __name__ == '__main__':


    img = cv2.imread('./archive/images/maksssksksss0.png')
    _, regions = selective_search(img,scale=400,min_size=10)
    print(type(regions),len(regions))
    print(regions[0])
    print(regions[1])
    print(regions[0]['rect'])


    _,bnd_box = utils.read_xml('./archive/annotations/maksssksksss0.xml')

    print(bnd_box)
    pos, neg=show_iou(bnd_box,regions,0.5,0.5)
    print(pos)
    print(neg)
    # show_region_proposal(img,regions,1000)

    # anno_files = utils.search_file('annotations')
    #
    # for ann in anno_files:
    #     og,_ = utils.read_xml(ann)
    #     max1 = 0
    #     max2 = 0
    #     if og[0] > max1:
    #         max1 = og[0]
    #     if og[1] > max2:
    #         max2 = og[1]
    # print(max1, max2)

