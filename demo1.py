import os
import json
import cv2
import numpy as np
PATH = 'E:\\Datasets\\balloon\\balloon\\train'
SAVE_PATH = 'E:\\Datasets\\balloon\\balloon\\trainAnno'


def get_ballon_data():
    file = os.path.join(PATH, 'via_region_data.json')
    with open(file) as f:
        annos = json.load(f)
    for key in annos:
        raw = annos[key]
        # 指示文件名
        filename = raw['filename']
        img = cv2.imread(os.path.join(PATH, filename))
        # regions指示气球的GTBox, 并且是多边形标注而非简单方框, regions的元素数量就是图像中气球的数量
        regions = raw['regions']
        # 可视化测试
        for region in regions:
            px = regions[region]['shape_attributes']['all_points_x']
            py = regions[region]['shape_attributes']['all_points_y']
            cv2.rectangle(img, (min(px), min(py)), (max(px), max(py)), (255, 0, 0), 2)
        # cv2.imwrite(os.path.join(SAVE_PATH, filename), img)


if __name__ == '__main__':
    get_ballon_data()