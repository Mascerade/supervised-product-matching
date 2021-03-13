import pandas as pd
import os
import numpy as np
import random
from tqdm import tqdm
from src.preprocessing import remove_stop_words, randomize_units, replace_space_df
from src.common import create_final_data, Common

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

def gen_spec_combos():
    '''
    Generates combinations of the spec data (WARNING: THIS TAKES A VERY LONG TIME AND YOU MUST HAVE AT LEAST 16GB RAM TO DO THIS)
    '''

    combos = np.meshgrid(*[list(LaptopAttributes.cpu.keys()), LaptopAttributes.hard_drive, LaptopAttributes.ram])
    combos = np.array(combos).T.reshape(-1, 3)
    np.random.shuffle(combos)
    df = pd.DataFrame(data=combos, columns=['cpu', 'hard_drive', 'ram'])
    df.to_csv('data/train/spec_data_no_brand.csv')

    
def concatenate_row(row):
    '''
    Creates a string out of the row of product attributes (so row is a Pandas DataFrame)
    '''
    
    # Split the brand
    row['company'] = row['brand'].split(' ')[0]
    row['product'] = ' '.join(row['brand'].split(' ')[1:])
    
    # Create dictionary for drive types
    drive_options = {'ssd': Common.SSD_TYPES, 'hdd': Common.HARD_DRIVE_TYPES}

    # Special tags at the end of the amount of inches of the laptop and the RAM to simulate real data
    inch_attr = str(row['inches']) + random.choice([' inch', '"'])
    ram_attr = row['ram'] + random.choice([' ram', ' memory'])

    # This modifies the CPU attribute to sometimes have different types of elements to add some difference
    # Ex: Intel Core i7 7700k vs Core i7 7700k 4 Core 4.2 GHz CPU (Something like that)
    cpu_attr = row['cpu']
    cores = LaptopAttributes.cpu[cpu_attr][0]
    ghz = LaptopAttributes.cpu[cpu_attr][1]
    
    if random.random() > 0.5:
        cpu_attr = cpu_attr.split(' ')
        if random.choice([0, 1]):
            if 'Intel' in cpu_attr:
                cpu_attr.remove('Intel')
        if random.choice([0, 1]):
            if 'Core' in cpu_attr:
                cpu_attr.remove('Core')
        if random.choice([0, 1]):
            if 'AMD' in cpu_attr:
                cpu_attr.remove('AMD')
    
        cpu_attr = ' '.join(cpu_attr)
    
    # Random chance of putting the cores in the CPU attribute
    if cores != None:
        if random.random() > 0.7:
            cpu_attr = '{} {} {}'.format(cpu_attr, cores, 'Core')
    
    # Random chance of putting the GHz in the CPU attribute
    if random.random() > 0.7:
        cpu_attr = '{} {}'.format(cpu_attr, ghz)
    
    if random.random() > 0.55:
        cpu_attr = '{} {}'.format(cpu_attr, 'CPU')
    
    # Have random chance of getting rid of company or product attribute
    product_removed = False
    if (random.random() > 0.5):
        row['product'] = ''
        product_removed = True
    
    if (random.random() > 0.5 and not product_removed):
        row['company'] = ''
    
    body_modifier = ''
    if (random.random() < 0.6):
        body_modifier = random.choice(Common.BODY_ADD_INS)

    begin_modifier = ''
    if (random.random() < 0.6):
        begin_modifier = random.choice(Common.MODIFIERS)

    screen_modifier = ''
    if (random.random() < 0.6):
        screen_modifier = random.choice(Common.SCREEN_MODIFIERS)
    
    end_modifiers = []
    if (random.random() < 0.6):
        end_modifiers = random.sample(Common.END_ADD_INS, random.randint(3, 6))

    
    # Create a list for all the product attributes
    order_attrs = [begin_modifier,
                   row['company'],
                   row['product'],
                   body_modifier,
                   inch_attr,
                   screen_modifier
                  ]

    # Have a chance of adding "laptop" to the title
    if random.random() > 0.45:
        order_attrs.append('laptop')
    
    # Add the type of drive the hard drive attribute
    row['hard_drive'] = row['hard_drive'] + ' ' + random.choice(drive_options[row['drive_type']])
    
    spec_attrs = [row['hard_drive'],
                  # row['screen'],
                   cpu_attr,
                   ram_attr
                 ]
    
    # Shuffle only the attributte
    random.shuffle(spec_attrs)
    order_attrs = order_attrs + spec_attrs + end_modifiers
    
    return ' '.join(order_attrs)

def format_laptop_row(row, brand, inches, screen, drive_type):
    row['brand'] = brand
    row['inches'] = inches
    row['screen'] = screen
    row['drive_type'] = drive_type
    return row

def create_pos_neg_data(df, neg_attrs):
    temp = []
    for idx in tqdm(range(0, int(len(df) * 0.04))):
        # Must start off with two positive titles
        first_row = df.iloc[idx]
        neg_attr = neg_attrs[idx % len(neg_attrs)]
        
        # Randomly choose the attributes that are not already in the row
        brand = random.choice(LaptopAttributes.laptop_brands)
        inches = random.choice(list(LaptopAttributes.inches))
        screen = random.choice(list(LaptopAttributes.screen))
        drive_type = random.choice(['ssd', 'hdd'])
        
        pos = format_laptop_row(first_row.copy(), brand, inches, screen, drive_type)
        
        new_attr = pos[neg_attr]
        
        while new_attr == pos[neg_attr]:
            new_attr = random.sample(LaptopAttributes.get_all_data()[neg_attr.lower()], 1)[0]
        
        neg = pos.copy()
        neg[neg_attr] = new_attr

        temp.append([remove_stop_words(concatenate_row(pos.copy())), remove_stop_words(concatenate_row(pos.copy())), 1])
        temp.append([remove_stop_words(concatenate_row(pos.copy())), remove_stop_words(concatenate_row(neg.copy())), 0])
    
    return pd.DataFrame(temp, columns=Common.COLUMN_NAMES)

def create_laptop_data():
    '''
    If spec_data.csv has not been created, we create that first.
    Afterwards, create the positive and negative spec data (just more laptop data) 
    and save it to spec_train_data.csv
    '''

    file_path = 'data/train/spec_train_data_new.csv'
    if not os.path.exists(file_path):
        print('Generating data for laptops . . . ')
        populate_spec()
        if not os.path.exists('data/base/spec_data_no_brand.csv'):
            print('Generating spec data combinations. WARNING: THIS WILL CONSUME RESOURCES AND TAKE A LONG TIME.')
            gen_spec_combos()
        spec_df = pd.read_csv('data/base/spec_data_no_brand.csv')
        final_laptop_df = create_pos_neg_data(spec_df, neg_attrs=['cpu', 'ram', 'inches', 'hard_drive'])
        final_laptop_df.reset_index(inplace=True)
        randomize_units(final_laptop_df, units=['gb'])
        final_laptop_df.to_csv(file_path)

    else:
        print('Already have spec data. Moving on . . .')