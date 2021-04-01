import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from stem import Signal
from stem.control import Controller
from tbselenium.tbdriver import TorBrowserDriver
import os
import time
import random
from pcpartpicker import switchIP

def amazon_laptop_title_collector():
    column_names = ['title']
    file_path = 'data/base/amazon_laptop_titles.csv'
    pages = 20
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=column_names)
    
    else:
        df = pd.read_csv(file_path)
    
    for page in range(int(pages)):
        soup = None
        switchIP()
        with TorBrowserDriver('/home/jason/.local/share/torbrowser/tbb/x86_64/tor-browser_en-US/') as driver:
            driver.get('https://www.amazon.com/s?k=laptop&page={}'.format(str(page + 1)))
            time.sleep(random.randint(13, 20))
            soup = BeautifulSoup(driver.page_source, 'lxml')
    
        for product in soup.find_all('h2', attrs={'class': 'a-size-mini a-spacing-none a-color-base s-line-clamp-2'}):
            try:
                title = product.find('a', attrs={'class': 'a-link-normal a-text-normal'}).find('span', {'class': 'a-size-medium a-color-base a-text-normal'}).text
                print('Title: ', title)
                df = df.append(pd.DataFrame([[title]], columns=column_names))

            except AttributeError as e:
                print(str(e))
        
    df.to_csv('/home/jason/Documents/Supervised-Product-Similarity/' + file_path)

def walmart_laptop_collector():
    column_names = ['title']
    if not os.path.exists('data/base/walmart_laptop_titles.csv'):
        df = pd.DataFrame(columns = column_names)
    
    else:
        df = pd.read_csv('data/base/walmart_laptop_titles.csv')
    
    for page in range(25):
        driver = webdriver.Chrome()
        driver.get('https://www.walmart.com/search/?page={}&query=laptops'.format(page + 1))
        soup = BeautifulSoup(driver.page_source, 'lxml')
        time.sleep(random.randint(8, 12))
        driver.quit()

        for product in soup.find_all('a', attrs={'class': 'product-title-link line-clamp line-clamp-2 truncate-title'}):
            try:
                title = product.find('span').text
                print("Title: {}".format(title))
                df = df.append(pd.DataFrame([[title]], columns=column_names))
            
            except AttributeError as e:
                print(str(e))
        
        df.to_csv('../../data/base/walmart_laptop_titles.csv')
        time.sleep(random.randint(5, 10))

if __name__ == "__main__":
    walmart_laptop_collector()
