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

intel_core_links = [
    'https://ark.intel.com/content/www/us/en/ark/products/series/79666/legacy-intel-core-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/94028/5th-generation-intel-core-m-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/94025/6th-generation-intel-core-m-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/95542/7th-generation-intel-core-m-processors.html'
    'https://ark.intel.com/content/www/us/en/ark/products/series/185341/8th-generation-intel-core-m-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/75025/4th-generation-intel-core-i3-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/84981/5th-generation-intel-core-i3-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/88394/6th-generation-intel-core-i3-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/95545/7th-generation-intel-core-i3-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/122588/8th-generation-intel-core-i3-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/134901/9th-generation-intel-core-i3-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/195733/10th-generation-intel-core-i3-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/202987/11th-generation-intel-core-i3-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/75024/4th-generation-intel-core-i5-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/84980/5th-generation-intel-core-i5-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/88393/6th-generation-intel-core-i5-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/95543/7th-generation-intel-core-i5-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/122597/8th-generation-intel-core-i5-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/134902/9th-generation-intel-core-i5-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/195732/10th-generation-intel-core-i5-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/202985/11th-generation-intel-core-i5-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/75023/4th-generation-intel-core-i7-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/84979/5th-generation-intel-core-i7-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/88392/6th-generation-intel-core-i7-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/95544/7th-generation-intel-core-i7-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/122593/8th-generation-intel-core-i7-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/134907/9th-generation-intel-core-i7-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/195734/10th-generation-intel-core-i7-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/202986/11th-generation-intel-core-i7-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/134928/8th-generation-intel-core-i9-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/186673/9th-generation-intel-core-i9-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/195735/10th-generation-intel-core-i9-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/202984/11th-generation-intel-core-i9-processors.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/202779/intel-core-processors-with-intel-hybrid-technology.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/123588/intel-core-x-series-processors.html'
]

pentium_links = [
    'https://ark.intel.com/content/www/us/en/ark/products/series/78132/legacy-intel-pentium-processor.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/77920/intel-pentium-processor-1000-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/71253/intel-pentium-processor-2000-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/75606/intel-pentium-processor-3000-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/89613/intel-pentium-processor-4000-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/76756/intel-pentium-processor-n-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/95590/intel-pentium-processor-j-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/77772/intel-pentium-processor-g-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/91594/intel-pentium-processor-d-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/128993/intel-pentium-silver-processor-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/129958/intel-pentium-gold-processor-series.html'
]

celeron_links = [
    'https://ark.intel.com/content/www/us/en/ark/products/series/79083/legacy-intel-celeron-processor.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/72000/intel-celeron-processor-1000-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/72017/intel-celeron-processor-2000-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/78948/intel-celeron-processor-3000-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/189306/intel-celeron-processor-4000-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/197892/intel-celeron-processor-5000-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/208610/intel-celeron-processor-6000-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/87282/intel-celeron-processor-n-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/77504/intel-celeron-processor-j-series.html',
    'https://ark.intel.com/content/www/us/en/ark/products/series/90613/intel-celeron-processor-g-series.html'
]

def amazon_laptop_title_collector():
    column_names = ['title']
    file_path = '../../data/base/amazon_laptop_titles.csv'
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
        
    df.to_csv(file_path, index=False)

def walmart_laptop_collector():
    """
    Scrapes Walmart for laptop titles
    Must browse on Walmart manually so you don't get captchas
    """
    
    column_names = ['title']
    if not os.path.exists('../../data/base/walmart_laptop_titles.csv'):
        df = pd.DataFrame(columns = column_names)
    
    else:
        df = pd.read_csv('../../data/base/walmart_laptop_titles.csv')
    
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
        
        df.to_csv('../../data/base/walmart_laptop_titles.csv', index=False)
        time.sleep(random.randint(5, 10))

def newegg_laptop_collector():
    column_names = ['title']
    if not os.path.exists('../../data/base/newegg_laptop_titles.csv'):
        df = pd.DataFrame(columns = column_names)
    
    else:
        df = pd.read_csv('../../data/base/newegg_laptop_titles.csv')
    
    for page in range(100):
        driver = webdriver.Chrome()
        driver.get('https://www.newegg.com/p/pl?d=laptops&page={}'.format(page + 1))
        soup = BeautifulSoup(driver.page_source, 'lxml')
        time.sleep(random.randint(8, 12))
        driver.quit()

        for product in soup.find_all('div', attrs={'class': 'item-cell'}):
            try:
                title = product.find('a', {'class': 'item-title'}).text
                print("Title: {}".format(title))
                df = df.append(pd.DataFrame([[title]], columns=column_names))
            
            except AttributeError as e:
                print(str(e))
        
        df.to_csv('../../data/base/newegg_laptop_titles.csv', index=False)
        time.sleep(random.randint(5, 10))

def intel_processor_collector(link):
    column_names = ['title']
    if not os.path.exists('../../data/base/intel_cpus.csv'):
        df = pd.DataFrame(columns = column_names)
    
    else:
        df = pd.read_csv('../../data/base/intel_cpus.csv')
    
    driver = webdriver.Chrome()
    driver.get(link)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    time.sleep(random.randint(5, 10))
    driver.quit()

    for product in soup.find_all('td', attrs={'class': 'ark-product-name ark-accessible-color component'}):
        try:
            cpu = product.find('a').text
            cpu = cpu.replace('®', ' ')
            cpu = cpu.replace('™', ' ')
            cpu = cpu.replace('  ', ' ')
            print("Title: {}".format(cpu))
            df = df.append(pd.DataFrame([[cpu]], columns=column_names))
        
        except AttributeError as e:
            print(str(e))
    
    df.to_csv('../../data/base/intel_cpus.csv', index=False)

def amd_processor_collector():
    column_names = ['title']
    if not os.path.exists('../../data/base/amd_cpus.csv'):
        df = pd.DataFrame(columns = column_names)
    
    else:
        df = pd.read_csv('../../data/base/amd_cpus.csv')
    
    driver = webdriver.Chrome()
    driver.get('https://en.wikipedia.org/wiki/List_of_AMD_Athlon_microprocessors')
    soup = BeautifulSoup(driver.page_source, 'lxml')
    time.sleep(random.randint(5, 10))
    driver.quit()

    for product in soup.find_all('th', attrs={'style': 'text-align:left;'}):
        try:
            cpu = product.text.split('[')[0]
            print("Title: {}".format(cpu))
            df = df.append(pd.DataFrame([[cpu]], columns=column_names))
        
        except AttributeError as e:
            print(str(e))
    
    df.to_csv('../../data/base/amd_cpus.csv', index=False)

if __name__ == "__main__":
    amd_processor_collector()
