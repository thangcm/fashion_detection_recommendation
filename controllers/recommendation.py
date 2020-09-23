import urllib.request
from urllib.parse import urlparse
from flask import request
from bs4 import BeautifulSoup
import requests
import selenium
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

options = Options()
options.headless = True


def recommend(path):
    """
    Find similar clothes on the Internet
    """
    link = get_image_search_link(path)
    img_sources = crawl_images(link)
    return img_sources


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

