import os
import cv2
import json
PATH = 'E:\\Datasets\\balloon\\balloon\\val'
SAVE_PATH = 'E:\\Datasets\\balloon\\balloon\\instances_val_balloon.json'


def generateCOCO():
    # 提取出原始标注数据
    file = os.path.join(PATH, 'via_region_data.json')
    with open(file) as f:
        annos = json.load(f)

    # COCO标注主要由images, categories, annotations这3个字段表示
    coco_dict = {}
    images, annotations, categories = [], [], []
    image_id = 1
    anno_id = 1

    categories.append({
        'id': 1,
        'name': 'balloon',
        'supercategory': 'balloon'
    })

    for key in annos:
        raw = annos[key]
        filename = raw['filename']
        height, width = cv2.imread(os.path.join(PATH, filename)).shape[:2]

        images.append({
            'file_name': filename,
            'height': height,
            'width': width,
            'id': image_id
        })

        regions = raw['regions']
        for region in regions:
            px = regions[region]['shape_attributes']['all_points_x']
            py = regions[region]['shape_attributes']['all_points_y']
            min_x, max_x = min(px), max(px)
            min_y, max_y = min(py), max(py)

            annotations.append({
                'id': anno_id,
                'image_id': image_id,
                'category_id': 1,
                # [x, y, w, h]
                'bbox': [min_x, min_y, max_x-min_x, max_y-min_y],
                'area': float((max_x-min_x) * (max_y-min_y)),
                'iscrowd': 0,
            })
            anno_id += 1
        image_id += 1

    coco_dict['images'] = images
    coco_dict['categories'] = categories
    coco_dict['annotations'] = annotations

    #保存结果
    json.dump(coco_dict, open(SAVE_PATH, 'w'))


if __name__ == '__main__':
    generateCOCO()