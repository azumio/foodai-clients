"""
Test latency of foodai 

Usage: 

    pip3 install -r requirements.txt

    USER_KEY=xxxxx ENDPOINT=https://api3.azumio.com/ python3 example.py 

"""

import httpx
import numpy as np
from PIL import Image
import io
import os
import urllib
import time
from multiprocessing import Pool
import json
from joblib import Memory


class FoodAI:
    def __init__(self, user_key) -> None:
        self.user_key = user_key
        self.endpoint = os.environ.get("ENDPOINT", "https://api3.azumio.com/")
        # self.endpoint = "http://35.238.124.118/"

    def connect(self):
        self.client = httpx.Client(http2=False)
        self.ping()

    def recognize(self, image, persistent=True):
        headers = {
            "Content-Type": "image/jpeg",
            "Accept-Encoding": "gzip"
        }

        if persistent:
            foods = self.client.post(os.path.join(self.endpoint, "v1/foodrecognition/full") + "?user_key=" +
                                     urllib.parse.quote(self.user_key) + "&top=1", content=image, headers=headers)
        else:
            foods = httpx.post(os.path.join(self.endpoint, "v1/foodrecognition/full") + "?user_key=" +
                               urllib.parse.quote(self.user_key) + "&top=1", content=image, headers=headers)

        j = foods.json()
        # print(j)
        if "is_food" not in j:
            raise BaseException("Failed request", j)

        return j

    def ping(self):
        pong = self.client.get(self.endpoint)
        print(pong.text[:100])


mem = Memory(".image_cache")


def download_image_(url):
    resp = httpx.get(url)
    image = Image.open(io.BytesIO(resp.content)).resize((544,544))
    out = io.BytesIO()
    image.save(out,"jpeg",quality=90)
    return out.getvalue()


download_image = mem.cache(download_image_)


def download_test_images():
    with open("test_images.json", "r") as f:
        image_links = json.load(f)

    pool = Pool(10)
    return list(pool.map(download_image, [photo["photos"][0]["href"].replace("=s0", "=s544") for photo in image_links["checkins"]]))


def sanitize(filename):
    for c in ["//", "/", ".", ":", "__"] * 3:
        filename = filename.replace(c, "_")
    return filename


user_key = os.environ["USER_KEY"]
foodai = FoodAI(user_key=user_key)


def recognize_one(image):
    foodai = FoodAI(user_key=user_key)
    start = time.time()
    result = foodai.recognize(image, persistent=False)
    end = time.time()
    latency = end - start

    return result, latency, image


def test():

    foodai.connect()
    print(foodai.endpoint)

    MAX_PARALLEL = 10 
    num_parallel = min(MAX_PARALLEL, int(os.environ.get("NUM_PARALLEL", 10)))
    pool = Pool(num_parallel)

    print("Downloading test images")
    test_images = download_test_images()
    print(f"Cached {len(test_images)} images")
    image_sizes = np.array([len(image) for image in test_images])/1000
    print(f"Image sizes kb {', '.join(image_sizes.astype(int).astype(str).tolist())}")

    print("Image size percentiles:")
    for per in range(0,110,10):
            print(
                f"    {per}%, {np.round(np.percentile(image_sizes,per))}kb")

    print(f"num_parallel={num_parallel}")

    for persistent in [False]:
        latency_list = []
        for i, (result, latency, image) in enumerate(pool.imap_unordered(recognize_one, test_images)):
            #print(f"latency {latency}")
            latency_list.append([time.time(), latency * 1000,len(image)])
        
        print("Latency percentiles:")
        for per in [50, 60, 70, 80, 90, 95, 99]:
            print(
                f"    {per}%, {np.round(np.percentile(np.array(latency_list)[:,1],per))}ms")

    np.savetxt("latency_" + sanitize(foodai.endpoint) +
               time.strftime("%Y%m%d-%H%M%S") + ".txt", np.array(latency_list).astype(int))


if __name__ == "__main__":
    test()
