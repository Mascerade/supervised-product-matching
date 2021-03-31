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

# signal TOR for a new connection (IP)
def switchIP():
    with Controller.from_port(port = 9051) as controller:
        time.sleep(5)
        controller.authenticate()
        controller.signal(Signal.NEWNYM)

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

def hard_drive_collector():
    column_names = ['name', 'capacity', 'type', 'form_factor']
    file_path = 'data/train/hard_drive_data.csv'
    pages = 25
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=column_names)
    
    else:
        df = pd.read_csv(file_path)
    
    for page in range(int(pages)):
        soup = None
        switchIP()
        with TorBrowserDriver('/home/jason/.local/share/torbrowser/tbb/x86_64/tor-browser_en-US/') as driver:
            driver.get('https://pcpartpicker.com/products/internal-hard-drive/#page={}'.format(str(page + 1)))
            time.sleep(random.randint(13, 20))
            soup = BeautifulSoup(driver.page_source, 'lxml')
    
        for product in soup.find_all('tr', attrs={'class': 'tr__product'}):
            try:
                name = product.find('div', attrs={'class': 'td__nameWrapper'}).find('p').text
                capacity = product.find('td', attrs={'class': 'td__spec td__spec--1'}).text.replace('Capacity', '')
                drive_type = product.find('td', attrs={'class': 'td__spec td__spec--3'}).text.replace('Type', '')
                form_factor = product.find('td', attrs={'class': 'td__spec td__spec--5'}).text.replace('Form Factor', '')
                print('Name: ', name, '| Capacity: ', capacity, '| Type: ', drive_type, '| Form Factor: ', form_factor)
                df = df.append(pd.DataFrame([[name, capacity, drive_type, form_factor]], columns=column_names))

            except AttributeError as e:
                print(str(e))
        
    df.to_csv('/home/jason/Documents/Supervised-Product-Similarity/' + file_path)


def video_card_collector():
    column_names = ['name', 'chipset', 'memory', 'core-clock']
    file_path = 'data/train/video-cards-data.csv'
    pages = 25
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=column_names)
    
    else:
        df = pd.read_csv(file_path)
    
    for page in range(int(pages)):
        soup = None
        switchIP()
        with TorBrowserDriver('/home/jason/.local/share/torbrowser/tbb/x86_64/tor-browser_en-US/') as driver:
            driver.get('https://pcpartpicker.com/products/video-card/#page={}'.format(str(page + 1)))
            time.sleep(random.randint(13, 20))
            soup = BeautifulSoup(driver.page_source, 'lxml')
    
        for product in soup.find_all('tr', attrs={'class': 'tr__product'}):
            try:
                name = product.find('div', attrs={'class': 'td__nameWrapper'}).find('p').text
                chipset = product.find('td', attrs={'class': 'td__spec td__spec--1'}).text.replace('Chipset', '')
                memory = product.find('td', attrs={'class': 'td__spec td__spec--2'}).text.replace('Memory', '')
                core_clock = product.find('td', attrs={'class': 'td__spec td__spec--3'}).text.replace('Core Clock', '')
                print('Name: ', name, '| Chipset: ', chipset, '| Memory: ', memory, '| Core Clock: ', core_clock)
                df = df.append(pd.DataFrame([[name, chipset, memory, core_clock]], columns=column_names))

            except AttributeError as e:
                print(str(e))
        
    df.to_csv('/home/jason/Documents/Supervised-Product-Similarity/' + file_path)

def get_links():
    part_type = input('What part type do you want (CPU, CPU cooler,  memory, internal hard drive, motherboard, video card, power supply, case)? ')
    pages = input('How many pages are there? ')
    file_name = input('What should the name of the file be? ')
    link_file = open('data/pcpartpicker_misc/{}.txt'.format(file_name), 'a')

    for page in range(int(pages)):
        soup = None
        switchIP()
        with TorBrowserDriver('/home/jason/.local/share/torbrowser/tbb/x86_64/tor-browser_en-US/') as driver:
            driver.get('https://pcpartpicker.com/products/{}/#page={}'.format(part_type, str(page + 1)))
            time.sleep(random.randint(13, 20))
            soup = BeautifulSoup(driver.page_source, 'lxml')

        for product in soup.find_all('tr', attrs={'class': 'tr__product'}):
            try:
                link_file.write('https://pcpartpicker.com' + product.find('td', attrs={'class': 'td__name'}).find('a')['href'] + '\n')

            except AttributeError as e:
                print(str(e))

    link_file.close()

def get_pos_data():
    # Gets postive examples of different ram titles from different retailers
    # Left off on Kingston HyperX Fury RGB 32 GB (line 224 of ram_links.txt)
    # Left off on Intel Core i3-3240 Dual-Core Processor 3.4 Ghz (line 106 of cpu_links.txt)
    file_name = input('What file do you want to open? ')
    csv_name = input('What would you like the finished CSV to be? ')
    link_file = open('data/pcpartpicker_misc/{}.txt'.format(file_name), 'r')
    retailer_names = ['amazon', 'bestbuy', 'newegg', 'walmart', 'memoryc', 'bhphotovideo']
    df = pd.DataFrame(columns=retailer_names)
    links = list(link_file)

    try:
        for link in links[51:]:
            link = link.strip()
            # Change the IP that Tor gives us
            switchIP()
            soup = None
            with TorBrowserDriver('/home/jason/.local/share/torbrowser/tbb/x86_64/tor-browser_en-US/') as driver:
                driver.get(link)
                time.sleep(15)
                soup = BeautifulSoup(driver.page_source, 'lxml')

            title_dict = dict(zip(retailer_names, ['' for x in range(len(retailer_names))]))

            for retailer in soup.find_all('td', attrs={'class': 'td__logo'}):
                link = 'https://www.pcpartpicker.com' + retailer.find('a')['href']
                soup = None
                for name in retailer_names:
                    if name in link:
                        if name == 'adorama':
                            # switchIP()
                            # with TorBrowserDriver('/home/jason/.local/share/torbrowser/tbb/x86_64/tor-browser_en-US/') as driver:
                            #     driver.get(link)
                            #     time.sleep(10)
                            #     soup = BeautifulSoup(driver.page_source, 'lxml')
                            # #soup = BeautifulSoup(driver.page_source, 'lxml')
                            # driver.quit()
                            pass

                        else:
                            driver = webdriver.Chrome()
                            driver.get(link)
                            soup = BeautifulSoup(driver.page_source, 'lxml')
                            driver.quit()
    
                        try:
                            if 'amazon' in link:
                                title_dict['amazon'] = soup.find('span', attrs={'id': 'productTitle'}).text.strip()
                                print('amazon', title_dict['amazon'])

                            elif 'bestbuy' in link:
                                title_dict['bestbuy'] = soup.find('h1', attrs={'class': 'heading-5 v-fw-regular'}).text.strip()
                                print('bestbuy', title_dict['bestbuy'])

                            elif 'newegg' in link:
                                title_dict['newegg'] = soup.find('h1', attrs={'id': 'grpDescrip_h'}).text.strip()
                                print('newegg', title_dict['newegg'])

                            elif 'walmart' in link:
                                title_dict['walmart'] = soup.find('h1', attrs={'class': 'prod-ProductTitle prod-productTitle-buyBox font-bold'}).text.strip()
                                print('walmart', title_dict['walmart'])
                            
                            elif 'memoryc' in link:
                                title_dict['memoryc'] = soup.find('section', attrs={'class': 'forCartImageItem'}).find('h1').text.strip()
                                print('memoryc', title_dict['memoryc'])
                            
                            elif 'bhphotovideo' in link:
                                title_dict['bhphotovideo'] = soup.find('h1', {'data-selenium': 'productTitle'}).text.strip()
                                print('bhphotovideo', title_dict['bhphotovideo'])

                            # elif 'adorama' in link:
                            #     title_dict['adorama'] = soup.find('div', attrs={'class': 'primary-info cf clear'}).find('h1').find('span').text.strip()
                            #     print('adorama', title_dict['adorama'])

                            else:
                                continue

                        except Exception:
                            pass
                
            df = df.append(pd.DataFrame([list(title_dict.values())], columns=retailer_names))

    except (Exception, KeyboardInterrupt) as e:
        print(str(e))

    print('here')
    df.to_csv('/home/jason/Documents/Supervised-Product-Similarity/data/train/{}.csv'.format(csv_name))
    link_file.close()

if __name__ == "__main__":
    video_card_collector()
