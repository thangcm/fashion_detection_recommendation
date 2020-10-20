import urllib.request
from urllib.parse import urlparse
from flask import request
from bs4 import BeautifulSoup
import requests
import selenium
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from utils import feature_extraction
from lshashpy3 import LSHash
import pickle
import numpy as np
from numpy.linalg import norm
import heapq

options = Options()
options.headless = True


def recommend(path):
    """
    Find similar clothes on the Internet
    """
    link = get_image_search_link(path)
    img_sources = crawl_images(link)
    return img_sources

def recommend_heapq(path):
    """
    Recommend based on cosine distance between the features of images
    Return the path of images which in top 5 results
    """
    features = pickle.load(open('feature_dict.p','rb'))
    img_feature = feature_extraction.feature_image(img_path=path)
    h = []
    for feature in features:
        cosine_distance = np.dot(feature["img_features"], img_feature)/(norm(img_feature)*norm(feature["img_features"]))
        heapq.heappush(h, (-cosine_distance, feature["img_path"]))

    out = []
    k = 5
    while k > 0:
        image = heapq.heappop(h)
        if image[0] < -0.8:
            print(image[0])
            out.append(image[1])
        k -= 1
    return out


def recommend_data(path):
    """
    Recommend images in data based on consine distance between the features vector of images
    Use locality sensitive hashing
    Return the path of the stored images
    """
    feature = feature_extraction.feature_image(img_path=path)
    lsh = LSHash(hash_size=10, input_dim=32768, num_hashtables=3,
        storage_config={ 'dict': None },
        matrices_filename='controllers/weights.npz',
        hashtable_filename='controllers/hash.npz'
        )
    
    results = lsh.query(feature, num_results=5, distance_func="cosine")
    return [a[0][1] for a in results]


def get_image_search_link(path):
    """
    Get image search link, after search by image
    """
    url = 'http://images.google.com/searchbyimage?image_url='
    host = urlparse(request.base_url).netloc
    url = url + host + '/' + path

    browser = webdriver.Firefox(options=options)
    browser.get(url)
    google_title = browser.find_element_by_tag_name('title-with-lhs-icon')
    contain_link_tag = google_title.find_element_by_tag_name('a')
    link = contain_link_tag.get_attribute('href')
    print(link)
    return link


def crawl_images(link):
    """
    Crawl all image in the first page of the search results
    """

    browser = webdriver.Firefox(options=options)
    browser.get(link)

    image_divs = browser.find_elements_by_class_name('isv-r')
    img_sources = []
    for image_div in image_divs:
        img_elements = image_div.find_element_by_tag_name('img')
        img_source = img_elements.get_attribute('src')
        img_sources.append(img_source)
    
    return img_sources

