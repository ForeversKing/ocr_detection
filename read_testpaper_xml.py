import glob, os
import cv2
import sys
import numpy as np
import xml.etree.ElementTree as ET
from setting import *

sys.path.append(PROJECT_ROOT)


def polygon_area(poly):
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.


def check_and_validate_polys(polys):
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    for poly in polys:
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
    return np.array(validated_polys)


def load_pascal_annoataion(p):
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)

    tree = ET.parse(p)
    anno = tree.getroot()
    objs = anno.findall('object')
    for obj in objs:
        rbox = obj.find("bndbox")
        xmin = float(rbox.find('xmin').text)
        xmax = float(rbox.find('xmax').text)
        ymin = float(rbox.find('ymin').text)
        ymax = float(rbox.find('ymax').text)
        text_polys.append([[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])
    return text_polys


xml_path = '../data/LikeVOC/Annotations'
# path = '/home/yulongwu/d/ocr/data/LikeVOC/Annotations'
text_lst = glob.glob(os.path.join(xml_path, '*xml'))
image_root = '../data/LikeVOC/JPEGImages'

for jj in text_lst:
    image_path = os.path.join(image_root, os.path.basename(jj).split('.')[0] +'.png')
    if not os.path.exists(image_path):
        image_path = os.path.join(image_root, os.path.basename(jj).split('.')[0] +'.jpg')
    print(image_path)
    img = cv2.imread(image_path)
    print('image shape:', img.shape)
    lines = load_pascal_annoataion(jj)
    data = check_and_validate_polys(lines)
    print('@@@@@@@@', lines)
    for k in lines:
        print(k)
        area = polygon_area(k)
        print(area)
        pts = np.array(k, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
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

