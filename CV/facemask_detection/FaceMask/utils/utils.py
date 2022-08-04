import cv2
import os
import xml.etree.ElementTree as ET

archive_path = 'C:\\Users\\ADMIN\\Desktop\\six\\Pycharm\\FaceMask\\archive'


def search_file(file):
    tmp = []
    filenames = os.listdir(archive_path + f'\\{file}')
    for filename in filenames:
        full_filename = os.path.join(archive_path + f'\\{file}', filename)
        tmp.append(full_filename)
    return tmp


def get_img_name(file):
    img_list = search_file(file)
    name_list = []
    for img in img_list:
        name_list.append(img.split('\\')[-1])
    return name_list


def reshape_img(size,file='images',):
    new_path = archive_path + f'\\reshape_{size}_img'
    os.mkdir(new_path)
    img_list = search_file(file)
    name_list = get_img_name(file)
    os.chdir(new_path)

    for img, name in zip(img_list, name_list):
        sample = cv2.imread(img)
        reshape_sample = cv2.resize(sample, (size, size))
        cv2.imwrite(new_path + '\\' + name, reshape_sample)
    print('Image Resize Finish !!')
    return new_path


def reshape_annotations(size, file='annotations',):
    new_path = archive_path + f'\\reshape_{size}_annotations'
    os.mkdir(new_path)
    os.chdir(new_path)

    anno_files = search_file(file)
    for anno in anno_files:
        og_size, bnd_list = read_xml(anno)
        name = anno.split('\\')[-1]
        name = name.split('.')[0]
        with open(f'{name}.csv','a') as f:
            for bnd in bnd_list:
                x_min = bnd[1] * size // og_size[0]
                y_min = bnd[2] * size // og_size[1]
                x_max = bnd[3] * size // og_size[0]
                y_max = bnd[4] * size // og_size[1]
                f.write(f"{bnd[0]}\t{x_min}\t{y_min}\t{x_max}\t{y_max}\n")
    print('Label Resize Finish !!')
    return new_path


def read_resized_annotation(size,resized_annotation_name):
    tmp = []
    with open(resized_annotation_name,'r') as f:
        for line in f:
            line = line.replace('\n','')
            line = line.split('\t')
            tmp.append(list(map(int,line)))
    return tmp


def read_xml(annotation_path):
    tree = ET.parse(annotation_path)
    size = tree.find('size')
    og_tmp = []
    for i in size:
        og_tmp.append(int(i.text))

    bnd_tmp = []
    ob = tree.findall('object')
    for o in ob:
        bnd = o.find('bndbox')
        name = o.find('name')
        if name.text == 'without_mask':
            tmp = [1]
        elif name.text == 'with_mask':
            tmp = [2]
        elif name.text == 'mask_weared_incorrect':
            tmp = [3]
        else:
            tmp [0]
        for b in bnd:
            tmp.append(int(b.text))
        bnd_tmp.append(tmp)
    return og_tmp, bnd_tmp


def img_with_bnd(img_path, bnd_list):
    img = cv2.imread(img_path)
    for bnd in bnd_list:
        if bnd[0] == 0:
            cv2.rectangle(img, (bnd[1], bnd[2]), (bnd[3], bnd[4]), (0, 0, 255), 0)
        else:
            cv2.rectangle(img, (bnd[1], bnd[2]), (bnd[3], bnd[4]), (0, 255, 0), 0)
    cv2.imshow('img',img)
    cv2.waitKey(0)


def img_with_bnd_after_dataloader(img,bnd_list):
    for bnd in bnd_list:
        if bnd[0] == 0:
            cv2.rectangle(img, (bnd[1], bnd[2]), (bnd[3], bnd[4]), (0, 0, 255), 0)
        else:
            cv2.rectangle(img, (bnd[1], bnd[2]), (bnd[3], bnd[4]), (0, 255, 0), 0)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def make_target(bnd_list):
    targets=[]
    for bnd in bnd_list:
        tmp = {}
        tmp['boxes'] = bnd[1:]
        tmp['labels'] = bnd[0]
        targets.append(tmp)
    return targets


if __name__ == '__main__':

    img_files = search_file('images')
    anno_files = search_file('annotations')
    print(img_files[0])
    og_size, bnd_list = read_xml(anno_files[0])
    print(og_size)
    print(bnd_list)

    # img_with_bnd(img_files[0],bnd_list)

    # reshape_annotations(size=256)

    # bnd_box=read_resized_annotation(256,'./archive/reshape_256_annotations/maksssksksss0.csv')
    # print(bnd_box)
    # a=make_target(bnd_box)
    # print(a)
    # resized_img_files = search_file('reshape_img')
    # img_with_bnd(resized_img_files[0],bnd_box)
