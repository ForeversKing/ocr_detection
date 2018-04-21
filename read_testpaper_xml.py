import glob, os
import cv2
import csv
import numpy as np
import xml.etree.ElementTree as ET
# path = '/home/yulongwu/d/east_data/east_icdar2015_resnet_v1_50_rbox/model.ckpt-49491.data-00000-of-00001'
# path = '/home/yulongwu/d/VOC2007/Annotations'
path = '/home/yulongwu/d/ocr/data/LikeVOC/Annotations'
text_lst = glob.glob(os.path.join(path, '*xml'))
image_root = '/home/yulongwu/d/VOC2007/JPEGImages'


def load_pascal_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)

    tree = ET.parse(p)
    anno = tree.getroot()
    objs = anno.findall('object')
    for obj in objs:
        rbox = obj.find("rotbox")
        x1 = float(rbox.find('ltx').text)
        y1 = float(rbox.find('lty').text)
        x2 = float(rbox.find('rtx').text)
        y2 = float(rbox.find('rty').text)
        x3 = float(rbox.find('rbx').text)
        y3 = float(rbox.find('rby').text)
        x4 = float(rbox.find('lbx').text)
        y4 = float(rbox.find('lby').text)
        text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return text_polys


for jj in text_lst:
    image_path = os.path.join(image_root, os.path.basename(jj).split('.')[0] +'.jpg')
    print(image_path)
    img = cv2.imread(image_path)
    print('image shape:', img.shape)
    lines = load_pascal_annoataion(jj)
    for k in lines:
        pts = np.array(k, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255), 1)
    cv2.imshow(' ', img)
    cv2.waitKey()

# for jj in text_lst:
#     image_label ='img_' + os.path.basename(jj).split('.')[0].split('_')[-1] + '.jpg'
#     image_path = os.path.join(path, image_label)
#     print(image_path)
#     img = cv2.imread(image_path)
#     fi = open(jj)
#     reader = csv.reader(fi)
#     lines = []
#     for line in reader:
#         line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
#         x1, y1, x2, y2, x3, y3, x4, y4 = list(line[:8])
#         lines.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
#     for k in lines:
#         pts = np.array(k, dtype=np.int32)
#         pts = pts.reshape((-1, 1, 2))
#         cv2.polylines(img, [pts], True, (0, 255, 255), 2)
#     cv2.imshow(' ', img)
#     cv2.waitKey()

