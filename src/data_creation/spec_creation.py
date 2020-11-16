import pandas as pd
import os
import numpy as np
import random
from tqdm import tqdm
from src.data_creation.laptop_data_creation import LaptopAttributes
from src.preprocessing import remove_stop_words
from src.common import create_final_data, modifiers, add_ins, hard_drive_types, ssd_types, COLUMN_NAMES

class SpecAttributes():
    """
    Different from LaptopAttributes, this is specific for creating spec data.
    The spec data was gathered from PCPartPicker and is used to create more laptop data.
    """
    video_card = {'GeForce RTX 2070'}
    ram = [str(x) + ' GB' for x in range(2, 130, 2)]
    hard_drive = [str(x) + ' GB' for x in range(120, 513, 8)] + [str(x) + ' TB' for x in range(1, 8)]
    cpu = {}
    laptop_brands = ['Lenovo ThinkPad', 'Lenovo ThinkBook', 'Lenovo IdeaPad', 'Lenovo Yoga', 'Lenovo Legion', 'HP Envy', 'HP Chromebook', 'HP Spectre', 'HP ZBook', 'HP Probook', 'HP Elitebook', 'HP Pavilion', 'HP Omen', 'Dell Alienware', 'Dell Vostro', 'Dell Inspiron', 'Dell Latitude', 'Dell XPS', 'Dell G Series', 'Dell Precision', 'Apple Macbook', 'Apple Macbook Air', 'Apple Mac', 'Acer Aspire', 'Acer swift', 'Acer Spin', 'Acer Switch', 'Acer Extensa', 'Acer Travelmate', 'Acer Nitro', 'Acer Enduro', 'Acer Predator', 'Asus ZenBook', 'Asus Vivobook', 'Asus Republic of Gamers', 'Asus ROG', 'Asus TUF GAMING']
    
    @staticmethod
    def get_all_data():
        return {
            'cpu': SpecAttributes.cpu.keys(),
            'ram': SpecAttributes.ram,
            'hard_drive': SpecAttributes.hard_drive,
            'video_card': SpecAttributes.video_card,
            'laptop_brands': SpecAttributes.laptop_brands
        }

def concatenate_spec_data(row):
    # Special tags at the end of the amount of inches of the laptop and the RAM to simulate real data
    inch_attr = str(row['inches']) + random.choice([' inch', '"'])
    ram_attr = row['ram'] + random.choice([' ram', ' memory'])

    # This modifies the CPU attribute to sometimes have different types of elements to add some difference
    # Ex: Intel Core i7 7700k vs Core i7 7700k 4 Core 4.2 GHz CPU (Something like that)
    cpu_attr = row['cpu']
    cores = SpecAttributes.cpu[cpu_attr][0]
    ghz = SpecAttributes.cpu[cpu_attr][1]
    
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
    if random.random() > 0.7:
        cpu_attr = '{} {} {}'.format(cpu_attr, cores, 'Core')
    
    # Random chance of putting the GHz in the CPU attribute
    if random.random() > 0.7:
        cpu_attr = '{} {}'.format(cpu_attr, ghz)
    
    if random.random() > 0.55:
        cpu_attr = '{} {}'.format(cpu_attr, 'CPU')
    
    # Create a list for all the product attributes
    order_attrs = [row['company'],
                   row['product'],
                   inch_attr,
                  ]

    # Have a chance of adding "laptop" to the title
    if random.random() > 0.45:
        order_attrs.append('laptop')
    
    spec_attrs = [row['hard_drive'],
                  # row['screen'],
                   cpu_attr,
                   ram_attr
                 ]
    
    # Shuffle only the attributte
    random.shuffle(spec_attrs)
    order_attrs = order_attrs + spec_attrs
    
    return ' '.join(order_attrs)

# Creates the negative examples for the laptop data
# The laptop_df is the original data, the new_df is the dataframe to append the new data to
# and the attributes are the attributes to swap for the new data
def create_neg_spec_laptop(df, attributes):
    df_iloc = df.iloc()
    temp = []
    for row in tqdm(range(int(len(df) * 1.91e-4))):
        # Create a copy of the row for the negative example
        for attribute_class in attributes:
            neg_row = df_iloc[row]
            # Get the row in the laptop_data and add the inch attribute
            orig_row = df_iloc[row]
            
            # Set product and company
            orig_row['company'] = orig_row['brand'].split(' ', 1)[0]
            orig_row['product'] = orig_row['brand'].split(' ', 1)[1]
            neg_row['company'] = orig_row['brand'].split(' ', 1)[0]
            neg_row['product'] = orig_row['brand'].split(' ', 1)[1]
            
            # Get a random inch attribute
            inch_attr = random.choice(list(LaptopAttributes.inches))
            
            # Get random screen attribute
            # screen_attr = random.choice(list(LaptopAttributes.screen))
            
            # Set the attributes
            orig_row['inches'] = inch_attr
            neg_row['inches'] = inch_attr
            # orig_row['screen'] = screen_attr
            # neg_row['screen'] = screen_attr
            
            if attribute_class == 'inches':
                # New inch attribute
                new_inch_attr = inch_attr

                # If the original attribute is still the same, keep getting a random one
                while inch_attr == new_inch_attr:
                    new_inch_attr = random.choice(list(LaptopAttributes.inches))
                
                neg_row['inches'] = new_inch_attr
            
            elif attribute_class == 'screen':
                # Optionally have a screen attribute (if it is part of the list of attributes to adjust)
                orig_screen_attr = random.choice(list(LaptopAttributes.screen))
                
                # New screen attribute
                new_screen_attr = screen_attr
                
                # If the original attribute is still the same, keep getting a random one
                while orig_screen_attr == new_screen_attr:
                    new_screen_attr = random.choice(list(LaptopAttributes.screen))
                
                neg_row['screen'] = new_screen_attr
                orig_row['screen'] = orig_screen_attr
            
            elif attribute_class == 'product':
                # New product attr
                new_product_attr = orig_row['product']
                
                # If the original attribute is still the same, keep getting a random one
                while orig_row['product'] == new_product_attr:
                    new_product_attr = random.choice(SpecAttributes.laptop_brands).split(' ', 1)[1]
                
                neg_row['product'] = new_product_attr
            
            elif attribute_class == 'hard_drive':
                # New drive attribute
                new_drive_attr = orig_row['hard_drive']
                
                # If the original attribute is still the same, keep getting a random one
                while orig_row['hard_drive'] == new_drive_attr:
                    new_drive_attr = random.choice(SpecAttributes.hard_drive)
                
                neg_row['hard_drive'] = '{} {}'.format(new_drive_attr, random.choice([random.choice(hard_drive_types), random.choice(ssd_types)]))
                orig_row['hard_drive'] = '{} {}'.format(orig_row['hard_drive'], random.choice([random.choice(hard_drive_types), random.choice(ssd_types)]))
            
            else:
                # Get the attribute that we are trying to change
                attribute_val = orig_row[attribute_class]

                # Temporarily value for the new value
                new_val = attribute_val

                # Make sure we really get a new attribute
                while new_val == attribute_val:
                    new_val = random.sample(SpecAttributes.get_all_data()[attribute_class.lower()], 1)[0]

                # Change the value in the neg_row to the new value
                neg_row[attribute_class] = new_val
            
            # We still need to add the phrasing to the hard drive attribute if it is not the current attribute class
            if attribute_class != 'hard_drive':
                drive_type = random.choice([random.choice(hard_drive_types), random.choice(ssd_types)])
                neg_row['hard_drive'] = '{} {}'.format(neg_row['hard_drive'], drive_type)
                orig_row['hard_drive'] = '{} {}'.format(orig_row['hard_drive'], drive_type)
            
            # Concatenate and normalize the data
            title_one = remove_stop_words(concatenate_spec_data(orig_row).lower())
            title_two = remove_stop_words(concatenate_spec_data(neg_row).lower())
            
            # Append the data to the temp list
            temp.append([title_one, title_two, 0])

    # Return the DataFrame created from temp
    return pd.DataFrame(temp, columns=COLUMN_NAMES)

# Creates the postive examples for the laptop data
# The laptop_df is the original data, the new_df is the dataframe to append the new data to
# and the attributes are the attributes to swap or delete for the new data
def create_pos_spec_data(df, rm_attrs, add_attrs):
    temp = []
    df_iloc = df.iloc()
    COLUMN_NAMES = ['title_one', 'title_two', 'label']
    for row in tqdm(range(int(len(df) * 2.8e-4))):
        # Set the new row to the same as the original to begin changing it
        new_row = df_iloc[row]

        # Get the row in the df and add the inch attribute
        orig_row = df_iloc[row]

        # Set product and company
        orig_row['company'] = orig_row['brand'].split(' ', 1)[0]
        orig_row['product'] = orig_row['brand'].split(' ', 1)[1]
        new_row['company'] = orig_row['brand'].split(' ', 1)[0]
        new_row['product'] = orig_row['brand'].split(' ', 1)[1]

        # Get a random inch attribute
        inch_attr = random.choice(list(LaptopAttributes.inches))

        # Get random screen attribute
        # screen_attr = random.choice(list(LaptopAttributes.screen))

        # Get random hard drive attribute and type
        hard_drive_attr = random.choice(list(SpecAttributes.hard_drive))
        
        # Get whether it will be an ssd or a hard drive
        drive_type = random.choice([hard_drive_types, ssd_types])

        # Set the attributes
        orig_row['inches'] = inch_attr
        # orig_row['screen'] = screen_attr

        orig_row['hard_drive'] = '{} {}'.format(hard_drive_attr, random.choice(drive_type))
        new_row['inches'] = inch_attr
        # new_row['screen'] = screen_attr
        new_row['hard_drive'] = '{} {}'.format(hard_drive_attr, random.choice(drive_type))
        
        for attr_list in rm_attrs:
            # Simply create a copy of new_row so that we do not have to keep on generating the same thing
            pos_row = new_row.copy()
            
            for attr in attr_list:
                pos_row[attr] = ''
        
            title_one = remove_stop_words(concatenate_spec_data(orig_row).lower())
            title_two = remove_stop_words(concatenate_spec_data(pos_row).lower())
    
            temp.append([title_one, title_two, 1])
    
    return pd.DataFrame(temp, columns=COLUMN_NAMES)

def populate_spec():
    # Getting the CPU data into SpecAttrbutes
    cpu_df = pd.read_csv('data/train/cpu_data.csv')
    temp_iloc = cpu_df.iloc()
    for idx in range(len(cpu_df)):
        row = temp_iloc[idx]
        SpecAttributes.cpu[row['name']] = [row['cores'], row['core_clock']]

    # Getting the video card data into SpecAttributes
    video_card_df = pd.read_csv('data/train/video-cards-data.csv')
    temp_iloc = video_card_df.iloc()
    for idx in range(len(video_card_df)):
        row = temp_iloc[idx]
        SpecAttributes.video_card.update([row['chipset']])

def gen_spec_combos():
    # Generates combinations of the spec data (WARNING: THIS TAKES A VERY LONG TIME AND YOU MUST HAVE AT LEAST 16GB RAM TO DO THIS)
    combos = np.meshgrid(*[SpecAttributes.laptop_brands, list(SpecAttributes.cpu.keys()), SpecAttributes.hard_drive, SpecAttributes.ram])
    combos = np.array(combos).T.reshape(-1, 4)
    np.random.shuffle(combos)
    df = pd.DataFrame(data=combos, columns=['brand', 'cpu', 'hard_drive', 'ram'])
    df.to_csv('data/train/spec_data.csv')

def create_spec_laptop_data():
    file_path = 'data/train/spec_train_data.csv'
    if not os.path.exists(file_path):
        print('Generating general spec data for laptops . . . ')
        populate_spec()
        if not os.path.exists('data/train/spec_data.csv'):
            print('Generating spec data combinations. WARNING: THIS WILL CONSUME RESOURCES AND TAKE A LONG TIME.')
            gen_spec_combos()
        spec_df = pd.read_csv('data/train/spec_data.csv')
        pos_df = spec_pos_df = create_pos_spec_data(spec_df, rm_attrs = [['company'], ['product']], add_attrs = [])
        neg_df = create_neg_spec_laptop(spec_df, ['cpu', 'ram', 'hard_drive', 'product', 'inches'])
        final_spec_df = create_final_data(pos_df, neg_df)
        print(len(final_spec_df))
        final_spec_df.to_csv(file_path)

    else:
        print('Already have spec data. Moving on . . .')