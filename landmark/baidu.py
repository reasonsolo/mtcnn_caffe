#coding=utf-8
import base64
import cv2
import json
import requests
import time
import pandas as pd
import re
import numpy as np


def get_baidu_token():
    request_url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={ak}&client_secret={sk}"
    app_key, app_secret = "GfL2aZlvtiIhflM3ZfkQ0gCK", "cOC17YcwkKbFo0CqjF1KGCU59uEIV5yO"
    rsp = requests.get(request_url.format(ak=app_key, sk=app_secret), timeout=3)
    access_token = rsp.json().get("access_token")
    return access_token

def web_exception_watch(default=None):
    def wrapper(func):
        def _wrapper(*args, **kargs):
            try:
                return func(*args, **kargs)
            except Exception as ee:
                print ("exception as ee", ee)
                return default
        return _wrapper
    return wrapper

'''
    "眼镜 大妈", "儿童流行头发", "儿童流行头发", "男生儿童流行头发", "男生少年流行头发", "少女头发图片"
    "大眼镜 男生", "眼镜 大爷", "戴眼镜的女生", "戴眼镜 女"
     "中年女性图片", "中年女性头发", "中年男性图片", "青年男生头发", "青年女性头发", "中国老年人头发", "没化妆的女性", "亚洲人脸", "亚洲人脸素材"
'''
QWORDS = ["小学生 脸部特写", "儿童 写真", "少年 写真", "小女孩 写真", "初中生 写真", "少女 写真"]
ptns = '''https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={query_words}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&word={query_words}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&pn={pic_id}&rn={read_nums}&gsm=12c&{timestamps}='''
hash_ptn = r"u=\d+,\d+&"
ACCESS_TOKEN = get_baidu_token()

def b64_image(img_mat):
    _, img_string = cv2.imencode('.jpg', img_mat)
    base64_str = base64.b64encode(img_string)
    return base64_str

def call_baidu_api(img_mat, url):
    # 调用百度的识别的api, face_field增加landmark
    time.sleep(0.4)
    result = []
    base64_str = b64_image(img_mat)
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect?access_token={0}".format(ACCESS_TOKEN)
    params = dict(image=str(base64_str), image_type="BASE64",
                  face_field="age,expression,gender,glasses,race,quality,landmark")
    rsp = requests.post(request_url, json=params, timeout=5)
    content = rsp.json() or {}
    face_lists = (content.get("result", {}) or {}).get("face_list", [])
    for idx, r in enumerate(face_lists):
        face_prob = r.get("face_probability", 0)
        if face_prob < 0.90:
            continue
        race = r.get("race", {}).get("type", "")
        age = int(r.get("age", 0))
        gender = r.get("gender", {}).get("type", "")
        beauty = int(r.get("beauty", 0))
        expression = r.get("expression", {}).get("type", "")
        landmarks = r.get("landmark72", [])
        result.append([url, idx, race, age, gender, beauty, expression, landmarks])
    return result

def test_all_words(words):
    query_map = {"urls": [], "hash_id": []}
    query_result = map(max_query, words)
    for idx, (urls, hash_ids) in enumerate(query_result):
        query_map["urls"] += urls
        query_map["hash_id"] += hash_ids
    data_frames = pd.DataFrame(query_map)
    data_frames = data_frames.drop_duplicates("hash_id")
    return data_frames

def max_query(word, max_nums=1000, read_nums=30):
    start_nu, crawled_urls, hash_ids = 0, [], []
    while(start_nu < max_nums):
        time.sleep(0.3)
        img_urls, start_nu, h_ids = run_query(word, start_nu, read_nums)
        crawled_urls += img_urls
        hash_ids += h_ids
        if start_nu % 10 == 0:
            print ("crawl idx:", start_nu)
        if len(img_urls) < read_nums:
           print("no data")
           break
    return (crawled_urls, hash_ids)

@web_exception_watch(([], -1, []))
def run_query(word, start_pic_num, read_nums):
    ptn = ptns.format(query_words=word, pic_id=start_pic_num, read_nums=read_nums, timestamps=int(time.time() * 1000))
    rsp = requests.get(ptn, timeout=3)
    img_urls, hash_ids = process_response(rsp)
    return img_urls, start_pic_num + read_nums, hash_ids

@web_exception_watch(([], []))
def process_response(rsp):
    img_urls, hash_ids = [], []
    data = (rsp.json() or {}).get("data", {})
    for elem in data:
        img_url = elem.get("middleURL")
        if img_url:
            img_urls.append(img_url)
            hash_ids.append(re.search(hash_ptn, img_url).group(0))
    return img_urls, hash_ids

