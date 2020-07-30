import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random

def ram_collector():
    column_names = ['name', 'speed']
    if not os.path.exists('data/train/ram_data.csv'):
        df = pd.DataFrame(columns = column_names)
    
    else:
        df = pd.read_csv('data/train/ram_data.csv')
    
    for page in range(75):
        driver = webdriver.Chrome()
        driver.get('https://pcpartpicker.com/products/memory/#page' + str(page + 1))
        soup = BeautifulSoup(driver.page_source, 'lxml')
        driver.quit()

        for product in soup.find_all('tr', attrs={'class': 'tr__product'}):
            try:
                name = product.find('div', attrs={'class': 'td__nameWrapper'}).find('p').text
                ram_speed = product.find('td', attrs={'class': 'td__spec td__spec--1'}).text.replace('Speed', '')
                df = df.append(pd.DataFrame([[name, ram_speed]], columns=column_names))

            except AttributeError as e:
                print(str(e))

    df.to_csv('data/train/ram_data.csv')

def cpu_collector():
    column_names = ['name', 'cores', 'core_clock']
    if not os.path.exists('data/train/cpu_data.csv'):
        df = pd.DataFrame(columns = column_names)
    
    else:
        df = pd.read_csv('data/train/cpu_data.csv')
    
    for page in range(13):
        driver = webdriver.Chrome()
        driver.get('https://pcpartpicker.com/products/cpu/#page=' + str(page + 1))
        soup = BeautifulSoup(driver.page_source, 'lxml')
        driver.quit()

        for product in soup.find_all('tr', attrs={'class': 'tr__product'}):
            try:
                name = product.find('div', attrs={'class': 'td__nameWrapper'}).find('p').text
                core_count = product.find('td', attrs={'class': 'td__spec td__spec--1'}).text.replace('Core Count', '')
                core_clock = product.find('td', attrs={'class': 'td__spec td__spec--2'}).text.replace('Core Clock', '')
                df = df.append(pd.DataFrame([[name, core_count, core_clock]], columns=column_names))
            
            except AttributeError as e:
                print(str(e))

    df.to_csv('data/train/cpu_data.csv')

def get_links():
    link_file = open('data/pcpartpicker_links/ram_links.txt', 'a')

    for page in range(75):
        driver = webdriver.Chrome()
        driver.get('https://pcpartpicker.com/products/memory/#page=' + str(page + 1))
        soup = BeautifulSoup(driver.page_source, 'lxml')
        time.sleep(1)
        driver.quit()

        for product in soup.find_all('tr', attrs={'class': 'tr__product'}):
            try:
                link_file.write('https://pcpartpicker.com' + product.find('td', attrs={'class': 'td__name'}).find('a')['href'] + '\n')

            except AttributeError as e:
                print(str(e))

    link_file.close()

def get_pos_data():
    link_file = open('data/pcpartpicker_links/ram_links.txt', 'r')

    for link in link_file:
        driver = webdriver.Chrome()
        driver.get(link)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        time.sleep(3)
        driver.quit()

        for retailer in soup.find_all('td', attrs={'class': 'td__logo'}):
            link = 'https://www.pcpartpicker.com' + retailer.find('a')['href']

            if 'amazon' in link:
                driver = webdriver.Chrome()
                driver.get(link)
                soup = BeautifulSoup(driver.page_source, 'lxml')
                driver.quit()

                title = soup.find('span', attrs={'id': 'productTitle'}).text.strip()
                print(title)

            if 'bestbuy' in link:
                driver = webdriver.Chrome()
                driver.get(link)
                soup = BeautifulSoup(driver.page_source, 'lxml')
                driver.quit()

                title = soup.find('h1', attrs={'class': 'heading-5 v-fw-regular'}).text.strip()
                print(title)

            if 'newegg' in link:
                driver = webdriver.Chrome()
                driver.get(link)
                soup = BeautifulSoup(driver.page_source, 'lxml')
                driver.quit()

                title = soup.find('h1', attrs={'id': 'grpDescrip_h'}).text.strip()
                print(title)

            if 'walmart' in link:
                driver = webdriver.Chrome()
                driver.get(link)
                soup = BeautifulSoup(driver.page_source, 'lxml')
                driver.quit()

                title = soup.find('h1', attrs={'class': 'prod-ProductTitle prod-productTitle-buyBox font-bold'}).text.strip()
                print(title)
            
            if 'memoryc' in link:
                driver = webdriver.Chrome()
                driver.get(link)
                soup = BeautifulSoup(driver.page_source, 'lxml')
                driver.quit()

                title = soup.find('section', attrs={'class': 'forCartImageItem'}).find('h1').text.strip()
                print(title)

    link_file.close()

get_pos_data()
