from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd

# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--proxy-server=socks5://localhost:9050')
# driver = webdriver.Chrome(options=chrome_options)
# driver.get('https://www.adorama.com/wd1tb3dssd.html?sterm=3YaykwzJ0xyORz2wUx0Mo38TUkiTUYV9RQlD1g0&utm_source=rflaid912925')
# sleep(5)
# soup = BeautifulSoup(driver.page_source, 'lxml')
# driver.quit()

# print('adorama', soup.find('div', attrs={'class': 'primary-info cf clear'}).find('h1').find('span').text.strip())

df = pd.DataFrame(columns=['junk', 'junk2'])
df = df.append(pd.DataFrame([[1, 2]], columns=['junk', 'junk2']))
df.to_csv('data/train/{}.csv'.format('junk'))
