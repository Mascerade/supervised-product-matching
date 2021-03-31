from stem import Signal
from stem.control import Controller
from tbselenium.tbdriver import TorBrowserDriver
from time import sleep

# signal TOR for a new connection
def switchIP():
    with Controller.from_port(port = 9051) as controller:
        sleep(5)
        controller.authenticate()
        controller.signal(Signal.NEWNYM)

for x in range(5):
    with TorBrowserDriver('/home/jason/.local/share/torbrowser/tbb/x86_64/tor-browser_en-US/') as driver:
        driver.get('https://www.pcpartpicker.com/products/cpu/')
        sleep(15)
        print(driver.page_source)
    switchIP()
