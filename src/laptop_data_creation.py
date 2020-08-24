import pandas as pd
from preprocessing import remove_stop_words
from common import create_final_data
import random

# This class will be used in order to exchange the different attributes
# to create negative examples
class Attributes():
    company = {'Apple'}
    product = {'MacBook Pro'}
    inches = {'13.3'}
    cpu = {'Intel Core i5 2.3GHz'}
    ram = {'4GB'}
    memory = {'256GB SSD'}
    gpu = {'Intel HD Graphics 520'}
    screen = {'1440x900'}
    
    @staticmethod
    def get_all_data():
        return {
            'company': Attributes.company,
            'product': Attributes.product,
            'inches': Attributes.inches,
            'cpu': Attributes.cpu,
            'ram': Attributes.ram,
            'memory': Attributes.memory,
            'gpu': Attributes.gpu,
            'screen': Attributes.screen
        }

# Create attribute sets
def create_attribute_sets(df):
    Attributes.company.update([row.Company for row in df[['Company']].itertuples()])
    Attributes.product.update([row.Product for row in df[['Product']].itertuples()])
    Attributes.inches.update([str(row.Inches) for row in df[['Inches']].itertuples()])
    Attributes.cpu.update([row.Cpu for row in df[['Cpu']].itertuples()])
    Attributes.ram.update([row.Ram for row in df[['Ram']].itertuples()])
    Attributes.memory.update([row.Memory for row in df[['Memory']].itertuples()])
    Attributes.gpu.update([row.Gpu for row in df[['Gpu']].itertuples()])
    Attributes.screen.update([row.ScreenResolution for row in df[['ScreenResolution']].itertuples()])

def concatenate_row(row):
    # Note: got rid of everything after the '(' because it has info about the actual specs of the laptop
    # so if we change the specs, we need to fix that too
    
    # Special tags at the end of the amount of inches of the laptop and the RAM to simulate real data
    inch_attr = str(row['Inches']) + random.choice([' inch', '', '"'])
    ram_attr = row['Ram'] + random.choice([' ram', ' memory', ''])
    
    # These are words that commonly come up with laptops
    modifiers = ['premium', 'new', 'fast', 'latest model']
    add_ins = ['USB 3.0', 'USB 3.1 Type-C', 'USB Type-C', 'Bluetooth', 'WIFI', 'Webcam', 'FP Reader',
               'HDMI', '802.11ac', '802.11 ac', 'home', 'flagship', 'business', 'GbE LAN', 'DVD-RW', 'DVD', 'Windows 10']
    
    cpu_attr = row['Cpu']
    if random.choice([0, 1]):
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

    # Create a list for all the product attributes
    order_attrs = [random.choice(modifiers),
                   row['Company'],
                   row['Product'].split('(')[0],
                   row['TypeName'],
                   inch_attr,
                   row['ScreenResolution'],
                   cpu_attr,
                   ram_attr,
                   row['Memory'],
                   row['Gpu']]
    
    order_attrs = order_attrs + random.sample(add_ins, random.choice([1, 2, 3, 4]))
    
    # Shuffle the data because in real data, it does not really matter what order the attributes are in
    random.shuffle(order_attrs)
    
    return ' '.join(order_attrs)

# Creates the negative examples for the laptop data
# The laptop_df is the original data, the new_df is the dataframe to append the new data to
# and the attributes are the attributes to swap for the new data
def create_neg_laptop_data(laptop_df, attributes):
    new_column_names = ['title_one', 'title_two', 'label']
    negative_df = pd.DataFrame(columns = new_column_names)
    for row in range(len(laptop_df)):
        # Create a copy of the row for the negative example
        neg_row = laptop_df.iloc[row]
        for attribute_class in attributes:
            # Get the row in the laptop_data
            orig_row = laptop_df.iloc[row]
            
            # Get the attribute that we are trying to change
            attribute_val = orig_row[attribute_class]
            
            # Temporarily value for the new value
            new_val = attribute_val
            
            # Make sure we really get a new attribute
            while new_val == attribute_val:
                new_val = random.sample(Attributes.get_all_data()[attribute_class.lower()], 1)[0]
            
            # Change the value in the neg_row to the new value
            neg_row[attribute_class] = new_val
            
            # Concatenate and normalize the data
            title_one = remove_stop_words(concatenate_row(orig_row).lower())
            title_two = remove_stop_words(concatenate_row(neg_row).lower())
            
            # Append the data to the new df
            negative_df = negative_df.append(pd.DataFrame([[title_one, title_two, 0]], columns=new_column_names))
    
    return negative_df

# Creates the postive examples for the laptop data
# The laptop_df is the original data, the new_df is the dataframe to append the new data to
# and the attributes are the attributes to swap or delete for the new data
def create_pos_laptop_data(laptop_df, rm_attrs, add_attrs):
    new_column_names = ['title_one', 'title_two', 'label']
    pos_df = pd.DataFrame(columns = new_column_names)
    for row in range(len(laptop_df)):
        # Remove the attribute from the new title
        for attr_list in rm_attrs:
            # Create a copy of the row for the negative example
            new_row = laptop_df.iloc[row]
            orig_row = laptop_df.iloc[row]
            for attr in attr_list:
                new_row[attr] = ''
        
            title_one = remove_stop_words(concatenate_row(orig_row).lower())
            title_two = remove_stop_words(concatenate_row(new_row).lower())

            pos_df = pos_df.append(pd.DataFrame([[title_one, title_two, 1]], columns=new_column_names))

    return pos_df

def create_laptop_data():
    # Load the laptop data
    laptop_df = pd.read_csv('data/train/laptops.csv', encoding='latin-1')
    create_attribute_sets(laptop_df)

    neg_df = create_neg_laptop_data(laptop_df, attributes=['Cpu', 'Memory', 'Ram', 'Inches', 'Product'])
    pos_df = create_pos_laptop_data(laptop_df, rm_attrs = [['Company'], ['TypeName'], ['ScreenResolution'], ['Product'], ['TypeName', 'ScreenResolution']], add_attrs = [])
    final_laptop_df = create_final_data(pos_df, neg_df)
    final_laptop_df = final_laptop_df.sample(frac=1)
    return final_laptop_df