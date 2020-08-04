from stem import Signal
from stem.control import Controller
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
from time import sleep

# signal TOR for a new connection
def switchIP():
    with Controller.from_port(port = 9051) as controller:
        sleep(5)
        controller.authenticate()
        controller.signal(Signal.NEWNYM)

# get a new selenium webdriver with tor as the proxy
def my_proxy(PROXY_HOST,PROXY_PORT):
    fp = webdriver.FirefoxProfile()
    # Direct = 0, Manual = 1, PAC = 2, AUTODETECT = 4, SYSTEM = 5
    fp.set_preference("network.proxy.type", 1)
    fp.set_preference("network.proxy.socks",PROXY_HOST)
    fp.set_preference("network.proxy.socks_port",int(PROXY_PORT))
    fp.update_preferences()
    options = Options()
    #options.headless = True
    return webdriver.Firefox(options=options, firefox_profile=fp)

for x in range(10):
    proxy = my_proxy("127.0.0.1", 9050)
    proxy.get("https://whatsmyip.com/")
    html = proxy.page_source
    soup = BeautifulSoup(html, 'lxml')
    print(soup.find("span", {"id": "ipv4"}))
    print(soup.find("span", {"id": "ipv6"}))
    switchIP()