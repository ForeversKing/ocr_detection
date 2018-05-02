import glob, os
import cv2
import csv
import numpy as np
import xml.etree.ElementTree as ET
# path = '/home/yulongwu/d/east_data/east_icdar2015_resnet_v1_50_rbox/model.ckpt-49491.data-00000-of-00001'
path = '/home/yulongwu/d/VOC2007/Annotations'
# path = '/home/yulongwu/d/east_data/icdar2015_train_data'
text_lst = glob.glob(os.path.join(path, '*xml'))
image_root = '/home/yulongwu/d/VOC2007/JPEGImages'

def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        print('!!!')
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if xmax - xmin < FLAGS.min_crop_side_ratio * w or ymax - ymin < FLAGS.min_crop_side_ratio * h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax + 1, xmin:xmax + 1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax + 1, xmin:xmax + 1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags


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
        text_tags.append(False)
    return text_polys, text_tags


for jj in text_lst:
    image_path = os.path.join(image_root, os.path.basename(jj).split('.')[0] +'.jpg')
    print(image_path)
    img = cv2.imread(image_path)
    print('image shape:', img.shape)
    lines = load_pascal_annoataion(jj)
    for k in lines:
        pts = np.array(k, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255), 3)
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
