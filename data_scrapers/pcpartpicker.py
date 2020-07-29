import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import os

df = pd.DataFrame(columns = ['title', 'speed'])
driver = webdriver.Chrome()
data = driver.get('https://pcpartpicker.com/products/memory/')
soup = BeautifulSoup(driver.page_source, 'lxml')
driver.quit()

for product in soup.find_all('tr'):
    try:
        name = product.find('div', attrs={'class': 'td__nameWrapper'}).find('p').text
        ram_speed = product.find('td', attrs={'class': 'td__spec td__spec--1'}).text.replace('Speed', '')
        df = df.append(pd.DataFrame([[name, ram_speed]], columns=['title', 'speed']))

    except AttributeError as e:
        print(str(e))

if not os.path.exists('data/train/ram_data.csv'):
    df.to_csv('data/train/ram_data.csv')
