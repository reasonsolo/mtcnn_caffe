import sys
import os


def iterate_wider(anno_path, functor):
    """
    functor(img_path, bboxes)
    """
    count = 0
    line_types = ["name", 'num', 'bbox']
    data = []
    with open(anno_path, "r") as f:
        line_type = 'name'
        num = 0
        bboxes = []
        img_path = None
        for line in f.readlines():
            if line_type == 'name':
                line_type = 'num'
                img_path = line.strip()
            elif line_type == 'num':
                num = int(line.strip())
                line_type = 'bbox'
            elif line_type == 'bbox':
                box = list(map(int, line.split()[0:4]))
                bbox = [box[0], box[1], box[2] + box[0], box[3] + box[1]]
                bboxes.append(bbox)
                if len(bboxes) == num:
                    functor(img_path, bboxes)
                    line_type = 'name'
                    bboxes = []


def iterate_lfw(anno_path, functor):
    """
    functor(img_path, bboxes, landm5)
    """
    data = []
    with open('anno_path', 'r') as f:
        for line in f:
            segs = line.split()
            img_name = segs[0]
            bbox = [int(x) for x in segs[1:5]]
            landm5 = [float(x) for x in segs[5:]]
            functor(img_name , bbox, landm5)
