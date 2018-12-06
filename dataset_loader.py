import sys
import os


def iterate_wider(anno_path, functor):
    """
    functor(img_path, bboxes)
    """
    count = 0
    line_types = ["name", 'num', 'bbox']
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
    with open(anno_path, 'r') as f:
        for line in f:
            segs = line.split()
            img_name = segs[0].replace('\\', '/')
            box = [int(x) for x in segs[1:5]]
            bbox = [box[0], box[2], box[1], box[3]]
            landm5 = [float(x) for x in segs[5:]]
            assert(len(landm5) == 10)
            try:
                functor(img_name, bbox, landm5)
            except Exception as ex:
                print(line)
                raise ex

def iterate_celeba_align(anno_path, functor):
    """
    functor(img_path, landm5)
    """
    with open(anno_path, 'r') as f:
        for line in f[2:]:
            segs = line.split()
            image_name = segs[0]
            landm5 = [float(x) for x in segs[1:]]
            assert(len(landm5) == 10)
            try:
                fucntor(img_name, landm5)
            except Exception as ex:
                print(line)
                raise ex


def iterate_300W(dataset_dir, functor):
    """
    functor(img_path, pts)
    """
    for root, dirs, files in os.walk("."):
        for pts_file in filter(lambda f: f.endswith('pts'), files):
            img_file = pts_file[:-3] + 'jpg'
            with open(pts_file, 'r') as ptsf:
                pts = []
                for line in ptsf[3:-1]:
                    pt = [float(x) for x in line.split()]
                    pts.append(tuple(pt))
            try:
                functor(img_path, pts)
            except Exception as ex:
                print(pts_file, img_file)
                raise ex

