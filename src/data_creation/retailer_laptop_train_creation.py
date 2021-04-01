import re
import math
import pandas as pd
import os
import random
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from src.data_creation.laptop_data_creation import LaptopAttributes, populate_spec
from src.preprocessing import remove_stop_words, unit_matcher, remove_misc
from src.common import create_final_data

populate_spec()
print(LaptopAttributes.inches)

"""" Set up sets """
laptop_brands = {'gateway', 'panasonic', 'toughbook', 'msi'}
cpu_attributes = {'intel'}

for brand in LaptopAttributes.laptop_brands:
    laptop_brands.add(brand.split(' ')[0].lower())
    laptop_brands.add(' '.join(brand.split(' ')[1: ]).lower())

intel_cpu_df = pd.read_csv('data/base/intel_cpus.csv')
intel_cpu_df = intel_cpu_df['title'].map(lambda x: remove_stop_words(x).split(' '))
for i in range(len(intel_cpu_df)):
    cpu_attributes.update(intel_cpu_df.iloc[i])

amd_cpu_df = pd.read_csv('data/base/amd_cpus.csv')
amd_cpu_df = amd_cpu_df['title'].map(lambda x: remove_stop_words(x).split(' '))
for i in range(len(amd_cpu_df)):
    cpu_attributes.update(amd_cpu_df.iloc[i])

laptop_brands = list(laptop_brands)
cpu_attributes = list(cpu_attributes)
matcher = unit_matcher('gb')

"""" Get dataframes """
amazon_laptops = pd.read_csv('data/base/amazon_laptop_titles.csv')
walmart_laptops = pd.read_csv('data/base/walmart_laptop_titles.csv')
newegg_laptops = pd.read_csv('data/base/newegg_laptop_titles.csv')

laptops = remove_misc(pd.concat([amazon_laptops, walmart_laptops, newegg_laptops]))
laptops['title'] = laptops['title'].apply(lambda x: remove_stop_words(x))
print(laptops)

re.compile('[1]+.{0,1}' + unit + '(?!\S)', re.IGNORECASE)

