import os
import sys
import cv2
import config
from baidu import call_baidu_api
sys.path.append('../')
from fast_mtcnn import fast_mtcnn
from dataset_loader import iterate_wider, iterate_lfw, iterate_300w, iterate_celeba

def expand_mtcnn_box(img, box):
    exp_ratio = 0.15
    h, w, _ = img.shape
    x1, y1, x2, y2 = box[:4]
    mid_x, mid_y =  (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    bw += bw * 0.15 * 2
    bh += bh * 0.15 * 2
    nx1, ny1 = max(0, mid_x - bw / 2), max(0, mid_y - bh/2)
    nx2, ny2 = min(mid_x + bw/2, w), max(0, mid_y + bh/2)
    return map(int, [nx1, ny1, nx2, ny2])

def is_valid_facebox(box):
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    return bw * bh > 1600

def extract_baidu_lm72(baidu_lm72):
    return [(pt["x"], pt["y"]) for pt in baidu_lm72]

def lfw_iter_func(lfw_dir, save_dir, mtcnn, anno_file, index, last_pic):
    continuation = {'skip': False}
    if lfw_dir in last_pic:
        continuation['skip'] = True
    def gen_data(img_name, bboxes, landm5):
        img_path = os.path.join(lfw_dir, img_name)
        if continuation['skip']:
            if img_path == last_pic:
                continuation['skip'] = False
            else:
                return
        print(img_path)
        img = cv2.imread(img_path)
        boxes = mtcnn(img_path)
        for box in boxes:
            if not is_valid_facebox(box):
                continue
            exp_box = expand_mtcnn_box(img, box)
            cropped = img[exp_box[1]:exp_box[3], exp_box[0]:exp_box[2]]
            h, w, _ = cropped.shape
            if h > 200 or w > 200:
                ratio = 200. / max(w, h)
                cropped = cv2.resize(cropped, (0, 0), fx=ratio, fy=ratio)
                print("resize img to %d %d" % (ratio * h, ratio * w))
            try:
                baidu_result = call_baidu_api(cropped, '')
            except:
                continue
            if len(baidu_result) == 1:
                index['count'] += 1
                landmarks = extract_baidu_lm72(baidu_result[0][-1])
                #for pt in landmarks:
                #    cv2.circle(cropped, (int(pt[0]), int(pt[1])), 1, (255, 255, 0), 1)
                save_img_path = os.path.join(save_dir, '%d.jpg' % index['count'])
                cv2.imwrite(save_img_path, cropped)
                landmarks_str = ' '.join([str(x) for pt in landmarks for x in pt])
                label = '%s %s' % (save_img_path, landmarks_str)
                anno_file.write(label)
                anno_file.write('\n')
            else:
                print("baidu result size %d, abort" % (len(baidu_result)))
        print("generate %d boxes for %s" % (len(boxes), img_path))
    return gen_data

def w300_iter_func(w300_dir, save_dir, mtcnn, anno_file, index, last_pic):
    continuation = {'skip': False}
    if w300_dir in last_pic:
        continuation['skip'] = True
    def gen_data(img_name, landm68):
        img_path = os.path.join(w300_dir, img_name)
        if continuation['skip']:
            if img_path == last_pic:
                continuation['skip'] = False
            else:
                return
        img = cv2.imread(img_path)
        boxes = mtcnn(img_path)
        for box in boxes:
            if not is_valid_facebox(box):
                continue
            exp_box = expand_mtcnn_box(img, box)
            cropped = img[exp_box[1]:exp_box[3], exp_box[0]:exp_box[2]]
            h, w, _ = cropped.shape
            if h > 200 or w > 200:
                ratio = 200. / max(w, h)
                cropped = cv2.resize(cropped, (0, 0), fx=ratio, fy=ratio)
                print("resize img to %d %d" % (ratio * h, ratio * w))
            try:
                baidu_result = call_baidu_api(cropped, '')
            except:
                continue
            if len(baidu_result) == 1:
                index['count'] += 1
                landmarks = extract_baidu_lm72(baidu_result[0][-1])
                #for pt in landmarks:
                #    cv2.circle(cropped, (int(pt[0]), int(pt[1])), 1, (255, 255, 0), 1)
                save_img_path = os.path.join(save_dir, '%d.jpg' % index['count'])
                cv2.imwrite(save_img_path, cropped)
                landmarks_str = ' '.join([str(x) for pt in landmarks for x in pt])
                label = '%s %s' % (save_img_path, landmarks_str)
                anno_file.write(label)
                anno_file.write('\n')
            else:
                print("baidu result size %d, abort" % (len(baidu_result)))
        print("generate %d boxes for %s" % (len(boxes), img_path))
    return gen_data

def celeba_iter_func(celeba_dir, save_dir, mtcnn, anno_file, index, last_pic):
    celeba_count = {'count': 0}
    continuation = {'skip': False}
    if w300_dir in last_pic:
        continuation['skip'] = True
    def gen_data(img_name, landm68):
        celeba_count['count'] += 1
        if celeba_count['count'] % 12 == 1:
            return None
        img_path = os.path.join(celeba_dir, img_name)
        if continuation['skip']:
            if img_path == last_pic:
                continuation['skip'] = False
            else:
                return
        img = cv2.imread(img_path)
        boxes = mtcnn(img_path)
        for box in boxes:
            if not is_valid_facebox(box):
                continue
            exp_box = expand_mtcnn_box(img, box)
            cropped = img[exp_box[1]:exp_box[3], exp_box[0]:exp_box[2]]
            h, w, _ = cropped.shape
            if h > 200 or w > 200:
                ratio = 200. / max(w, h)
                cropped = cv2.resize(cropped, (0, 0), fx=ratio, fy=ratio)
                print("resize img to %d %d" % (ratio * h, ratio * w))
            try:
                baidu_result = call_baidu_api(cropped, '')
            except:
                continue
            if len(baidu_result) == 1:
                index['count'] += 1
                landmarks = extract_baidu_lm72(baidu_result[0][-1])
                #for pt in landmarks:
                #    cv2.circle(cropped, (int(pt[0]), int(pt[1])), 1, (255, 255, 0), 1)
                save_img_path = os.path.join(save_dir, '%d.jpg' % index['count'])
                cv2.imwrite(save_img_path, cropped)
                landmarks_str = ' '.join([str(x) for pt in landmarks for x in pt])
                label = '%s %s' % (save_img_path, landmarks_str)
                anno_file.write(label)
                anno_file.write('\n')
            else:
                print("baidu result size %d, abort" % (len(baidu_result)))
        print("generate %d boxes for %s" % (len(boxes), img_path))
    return gen_data

def gen_rotate_img(img, box, pts, angle):
    pass


if __name__ == '__main__':
    save_dir = config.DATA_DIR % 'train'
    try:
        os.mkdir(save_dir)
    except:
        pass
    last_file = ''
    continuation = False
    index = {'count': 0}
    anno_path = os.path.join(save_dir, 'landmarks.txt')
    if len(sys.argv) > 1:
        count = 0
        last_file = sys.argv[1]
        with open(anno_path) as f:
            for line in f:
                count += 1
                file_path = line.split()[0]
                file_name = file_path.split('/')[-1]
                file_count = file_name.split('.')[0]
        index['count'] = int(file_count)
        continuation = True
    if len(sys.argv) > 2:
        continuation = True
        index['count'] = int(sys.argv[2])

    lfw_dir = os.path.join(config.LFW_DIR, 'lfw_train')
    lfw_anno_file = 'trainImageList.txt'
    lfw_anno_path = os.path.join(lfw_dir, lfw_anno_file)

    w300_dir = config.W300_DIR
    celeba_dir = config.CELEBA_DIR

    anno_file = open(os.path.join(save_dir, 'landmarks.txt'), 'a' if continuation else 'w')
    mtcnn = fast_mtcnn()

    #iterate_lfw(lfw_anno_path, lfw_iter_func(lfw_dir, save_dir, mtcnn, anno_file, index, last_file))
    #iterate_300w(w300_dir, w300_iter_func(w300_dir, save_dir, mtcnn, anno_file, index, last_file), False)
    iterate_celeba(celeba_dir, celeba_iter_func(celeba_dir, save_dir, mtcnn, anno_file, index, last_file))

    print(index)

