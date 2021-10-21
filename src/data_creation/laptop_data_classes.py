import pandas as pd
import re
from supervised_product_matching.model_preprocessing import remove_stop_words

class LaptopAttributes():
    '''
    Different from LaptopAttributes, this is specific for creating spec data.
    The spec data was gathered from PCPartPicker and is used to create more laptop data.
    '''

    video_card = {'GeForce RTX 2070'}
    ram = [str(x) + ' GB' for x in range(2, 130, 2)]
    hard_drive = [str(x) + ' GB' for x in range(120, 513, 8)] + [str(x) + ' TB' for x in range(1, 8)]
    cpu = {}
    laptop_brands = ['Lenovo ThinkPad', 'Lenovo ThinkBook', 'Lenovo IdeaPad', 'Lenovo Yoga', 'Lenovo Legion', 'HP Envy', 'HP Chromebook', 'HP Spectre', 'HP ZBook', 'HP Probook', 'HP Elitebook', 'HP Pavilion', 'HP Omen', 'Dell Alienware', 'Dell Vostro', 'Dell Inspiron', 'Dell Latitude', 'Dell XPS', 'Dell G Series', 'Dell Precision', 'Apple Macbook', 'Apple Macbook Air', 'Apple Mac', 'Acer Aspire', 'Acer Swift', 'Acer Spin', 'Acer Switch', 'Acer Extensa', 'Acer Travelmate', 'Acer Nitro', 'Acer Enduro', 'Acer Predator', 'Asus ZenBook', 'Asus Vivobook', 'Asus Republic of Gamers', 'Asus ROG', 'Asus TUF GAMING']
    screen = {'1440x900'}
    inches = {'13.3'}
    
    @staticmethod
    def get_all_data():
        return {
            'cpu': LaptopAttributes.cpu.keys(),
            'ram': LaptopAttributes.ram,
            'hard_drive': LaptopAttributes.hard_drive,
            'video_card': LaptopAttributes.video_card,
            'brand': LaptopAttributes.laptop_brands,
            'screen': LaptopAttributes.screen,
            'inches': LaptopAttributes.inches
        }

def populate_spec():
    '''
    Creates a string out of the row of product attributes (so row is a Pandas DataFrame).
    '''

    # Getting the CPU data into LaptopAttrbutes
    cpu_df = pd.read_csv('data/base/cpu_data.csv')
    temp_iloc = cpu_df.iloc()
    for idx in range(len(cpu_df)):
        row = temp_iloc[idx]
        LaptopAttributes.cpu[row['name']] = [row['cores'], row['core_clock']]

    # Getting the video card data into LaptopAttributes
    video_card_df = pd.read_csv('data/base/video-cards-data.csv')
    temp_iloc = video_card_df.iloc()
    for idx in range(len(video_card_df)):
        row = temp_iloc[idx]
        LaptopAttributes.video_card.update([row['chipset']])
    
    # Getting the inches, screen, video card, and CPU data from laptops.csv
    laptops_df = pd.read_csv('data/base/laptops.csv', encoding='latin-1')
    LaptopAttributes.inches.update([str(row.Inches) for row in laptops_df[['Inches']].itertuples()])
    LaptopAttributes.screen.update([row.ScreenResolution for row in laptops_df[['ScreenResolution']].itertuples()])
    LaptopAttributes.video_card.update([row.Gpu for row in laptops_df[['Gpu']].itertuples()]) 
    
    for row in laptops_df.iloc:
        if row.Company != 'Apple':
            LaptopAttributes.cpu[' '.join(row.Cpu.split(' ')[:-1])] = [None, row.Cpu.split(' ')[-1]]

class LaptopRetailerRegEx:
    populate_spec()
    laptop_brands = {'gateway', 'panasonic', 'toughbook', 'msi'}
    product_attrs = {'vivobook'}
    cpu_attributes = {'intel', 'm 2', '2 core', '4 core', '6 core', '8 core'}
    
    for brand in LaptopAttributes.laptop_brands:
        laptop_brands.add(brand.split(' ')[0].lower())
        product_attrs.add(' '.join(brand.split(' ')[1: ]).lower())

    intel_cpu_df = pd.read_csv('data/base/intel_cpus.csv')
    intel_cpu_df = intel_cpu_df['title'].map(lambda x: remove_stop_words(x, omit_punctuation=['.']).split(' '))
    for i in range(len(intel_cpu_df)):
        cpu_attributes.update(intel_cpu_df.iloc[i])

    amd_cpu_df = pd.read_csv('data/base/amd_cpus.csv')
    amd_cpu_df = amd_cpu_df['title'].map(lambda x: remove_stop_words(x, omit_punctuation=['.']).split(' '))
    for i in range(len(amd_cpu_df)):
        cpu_attributes.update(amd_cpu_df.iloc[i])

    laptop_brands = list(laptop_brands)
    laptop_brands.sort(key=len, reverse=True)

    product_attrs = list(product_attrs)
    product_attrs.sort(key=len, reverse=True)

    cpu_attributes = list(cpu_attributes)
    cpu_attributes.sort(key=len, reverse=True)

    ram_modifiers = ['memory', 'ram', 'ddr4', 'ddr4 ram', 'ddr4 memory']
    ram_modifiers.sort()

    hard_drive_modifiers = ['hdd', 'hard drive', 'disk drive', 'storage', 'hard drive storage', 'hdd storage']
    hard_drive_modifiers.sort(key=len, reverse=True)

    ssd_modifiers = ['ssd', 'solid state drive', 'solid state disk', 'pcie', 'pcie ssd', 'ssd storage']
    ssd_modifiers.sort(key=len, reverse=True)

    annoying_words = ['windows 10', 'win 10', 'windows 10 in s mode', 'windows', '3.0', '3.1', '3.2', 'optical drive', 'cd drive', 'dvd drive']
    annoying_words.sort(key=len, reverse=True)

    ram_modifier_matcher = re.compile("\\b" + "(?!\S)|\\b".join(ram_modifiers) + "(?!\S)", re.IGNORECASE)
    random_matcher = re.compile("\\b" + "(?!\S)|\\b".join(annoying_words) + "(?!\S)", re.IGNORECASE)
    cpu_matcher = re.compile("\\b" + "(?!\S)|\\b".join(cpu_attributes) + "(?!\S)", re.IGNORECASE)
    brand_matcher = re.compile("\\b" + "(?!\S)|\\b".join(laptop_brands) + "(?!\S)", re.IGNORECASE)
    product_attr_matcher = re.compile("\\b" + "(?!\S)|\\b".join(product_attrs) + "(?!\S)", re.IGNORECASE)
    ram_matcher = re.compile(' ?[0-9]+.{0,1}' + 'gb ?' + '(?:' + '|'.join([x for x in ram_modifiers]) + ')(?!\S)', re.IGNORECASE)
    hard_drive_matcher = re.compile(' ?[0-9]+.{0,1}' + '(?:gb|tb) ?' + '(?:' + '|'.join([x for x in hard_drive_modifiers]) + ')(?!\S)', re.IGNORECASE)
    ssd_matcher = re.compile(' ?[0-9]+.{0,1}' + '(?:gb|tb) ?' + '(?:' + '|'.join([x for x in ssd_modifiers]) + ')(?!\S)', re.IGNORECASE)
    gbtb_matcher = re.compile(' ?[0-9]+.{0,1}' + '(?:gb|tb)' + '(?!\S)', re.IGNORECASE)
    inch_matcher = re.compile('[1][0-9]\"?\"?.?[0-9]?\"?\"? ?(?:inch)?(?!\S)', re.IGNORECASE)
    del laptop_brands, product_attrs, cpu_attributes, intel_cpu_df, amd_cpu_df
