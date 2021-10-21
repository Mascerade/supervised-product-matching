import pandas as pd
import os
import random
from itertools import combinations
from src.data_creation.laptop_data_classes import LaptopRetailerRegEx
from src.data_preprocessing import remove_misc
from supervised_product_matching.model_preprocessing import remove_stop_words
from src.common import create_final_data

def get_key_attrs(title:str) -> tuple:
    """
    Get each major attribute of a laptop
    """

    # Remove random words that may end up in the important identifiers
    random_words = set(LaptopRetailerRegEx.random_matcher.findall(title))
    for word in random_words:
        title = title.replace(word, '')

    brand = list(map(lambda x: x.strip(), LaptopRetailerRegEx.brand_matcher.findall(title)))
    product_attr = list(map(lambda x: x.strip(), LaptopRetailerRegEx.product_attr_matcher.findall(title)))
    inch = list(map(lambda x: x.strip(), LaptopRetailerRegEx.inch_matcher.findall(title)))
    cpu = list(map(lambda x: x.strip(), LaptopRetailerRegEx.cpu_matcher.findall(title)))
    ram = list(map(lambda x: x.strip(), LaptopRetailerRegEx.ram_matcher.findall(title)))
    ssd = list(map(lambda x: x.strip(), LaptopRetailerRegEx.ssd_matcher.findall(title)))
    hard_drive = list(map(lambda x: x.strip(), LaptopRetailerRegEx.hard_drive_matcher.findall(title)))

    # Before getting the other gb attributes, make sure we don't get ones from ssd, hard drive or ram
    for x in ram:
        title = title.replace(x, '')
    
    for x in ssd:
        title = title.replace(x, '')

    for x in hard_drive:
        title = title.replace(x, '')

    other_gb_attrs = list(map(lambda x: x.strip(), LaptopRetailerRegEx.gbtb_matcher.findall(title)))

    return (brand, product_attr, inch, cpu, ram, ssd, hard_drive, other_gb_attrs)

def get_filler_tokens(orig_title: list, imp_tokens: list) -> list:
    """
    Get all of the filler words (words that are not major attributes)
    """

    filler_tokens = []
    for token in orig_title:
        if token not in imp_tokens:
            filler_tokens.append(token)

    return filler_tokens

def remove_filler_tokens(orig_title: list, filler_tokens: list) -> list:
    """"
    Generates new titles with less filler words in it
    """
    
    new_titles = []
    amt_filler_tokens = len(filler_tokens)
    if (len(filler_tokens) > 1):
        for x in range(len(filler_tokens)): # For as many filler tokens are there are, we are going to create that many new titles
            new_title = orig_title.copy()
            filler_tokens_cp = filler_tokens.copy()
            amt_to_remove = random.randint(int(amt_filler_tokens * 0.25), amt_filler_tokens)
            for x in range(amt_to_remove): # Get a random amount of filler tokens to remove
                filler = random.choice(filler_tokens_cp)
                new_title.remove(filler)
                filler_tokens_cp.remove(filler)
            new_titles.append(' '.join(new_title))
    
    return new_titles

def manipulate_ram(attr: str) -> str:
    """
    Uses different ways of saying ram
    """

    attr = attr.split('gb')[0] + 'gb ' + random.choice(LaptopRetailerRegEx.ram_modifiers)
    return attr

def manipulate_ssd(attr: str) -> str:
    """
    Uses different ways of saying an ssd
    """

    if 'gb' in attr:
        type_drive = 'gb '
    else:
        type_drive = 'tb '

    attr = attr.split(type_drive)[0] + type_drive + random.choice(LaptopRetailerRegEx.ssd_modifiers)
    return attr

def manipulate_hard_drive(attr: str) -> str:
    """
    Uses different ways of saying hard drive
    """

    if 'gb' in attr:
        type_drive = 'gb '
    else:
        type_drive = 'tb '

    attr = attr.split(type_drive)[0] + type_drive + random.choice(LaptopRetailerRegEx.hard_drive_modifiers)
    return attr

def manipulate_title_gbtb(titles: list, ssd_attrs: list, hard_drive_attrs: list, ram_attrs: list) -> str:
    """
    Uses the "manipulate" functions to vary the titles
    """

    modified_titles = []
    for x in titles:
        for drive in ssd_attrs:
            x = x.replace(drive, manipulate_ssd(drive))
        for drive in hard_drive_attrs:
            x = x.replace(drive, manipulate_hard_drive(drive))
        for mem in ram_attrs:
            x = x.replace(mem, manipulate_ram(mem))
        
        modified_titles.append(x)

    return modified_titles

def create_pos_laptop_data(df):
    """
    Using the scraped laptop data, create positive pairs
    """

    MAX_POS_TITLES = 6
    temp = []
    for title in df['title']:
        # Get each major attribute of a laptop
        brand, product_attr, inch, cpu, ram, ssd, hard_drive, other_gb_attrs = get_key_attrs(title)

        # Make sure the product is actually a laptop
        if len(ram) == 0 and len(ssd) == 0 and len(hard_drive) == 0 and len(other_gb_attrs) == 0:
            continue

        # Create a "simple" version of the title using only the major attributes
        shuffle = [cpu, *list(map(lambda x: x.split(' '), ram)),
        *list(map(lambda x: x.split(' '), ssd)), 
        *list(map(lambda x: x.split(' '), hard_drive)), 
        *list(map(lambda x: x.split(' '), other_gb_attrs))]
        random.shuffle(shuffle)

        pos_title1 = brand + product_attr + inch
        for x in shuffle:
            pos_title1 = pos_title1 + x

        # Get all of the filler words (words that are not major attributes)
        orig_title = title.split(' ')
        filler_tokens = get_filler_tokens(orig_title, pos_title1)

        # Generate a list of titles that do not have as many filler words
        new_titles = remove_filler_tokens(orig_title, filler_tokens)

        # Change up the less semantically meaningful attributes on drives/ram
        new_titles = manipulate_title_gbtb(new_titles, ssd, hard_drive, ram)
        
        # Choose how many combos we're going to have
        amt_new_titles = MAX_POS_TITLES
        if (len(new_titles) < MAX_POS_TITLES):
            amt_new_titles = len(new_titles)
        
        # Create the combination with the original title
        temp.append([title, ' '.join(pos_title1), 1])
        for x in range(amt_new_titles):
            pos = random.choice(new_titles)
            temp.append([title, pos, 1])
            new_titles.remove(pos)

        # Among the new titles, pair some of them up for more diversity
        combos = list(combinations(new_titles, 2))
        if (len(combos) > 4):
            ran_pairs = random.sample(combos, 4)
            for pair in ran_pairs:
                temp.append([pair[0], pair[1], 1])

    return pd.DataFrame(temp, columns=['title_one', 'title_two', 'label'])
        
def replace_drive_attribute(attr, ssd=False):
    """
    Replaces the drive attribute with a new one for negative data creation
    """

    gbs = [64, 128, 256, 484, 512, 768]
    tbs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if 'gb' in attr:
        type_drive = 'gb'
    else:
        type_drive = 'tb'
    
    orig_amt = int(attr.split(type_drive)[0].strip())
    
    if type_drive == 'gb':
        if orig_amt in gbs:
            gbs.remove(orig_amt)
        if ssd:
            return str(random.choice(gbs)) + random.choice([' ', '']) + 'gb ' + random.choice(LaptopRetailerRegEx.ssd_modifiers)
        else:
            return str(random.choice(gbs)) + random.choice([' ', '']) + 'gb ' + random.choice(LaptopRetailerRegEx.hard_drive_modifiers)
    
    else:
        if orig_amt in tbs:
            tbs.remove(orig_amt)
        if ssd:
            return str(random.choice(tbs)) + random.choice([' ', '']) + 'tb ' + random.choice(LaptopRetailerRegEx.ssd_modifiers)
        else:
            return str(random.choice(tbs)) + random.choice([' ', '']) + 'tb ' + random.choice(LaptopRetailerRegEx.hard_drive_modifiers)

def replace_ram_attribute(attr):
    """
    Replaces the ram attribute with a new one for negative data creation
    """
    
    gbs = [4, 8, 16, 24, 32, 48, 64]
    orig_amt = int(attr.split('gb')[0].strip())
    
    if orig_amt in gbs:
        gbs.remove(orig_amt)
    return str(random.choice(gbs)) + random.choice([' ', '']) + 'gb ' + random.choice(LaptopRetailerRegEx.ram_modifiers)

def replace_other_attribute(attr):
    """
    Replaces an "other" gb attribute with a new one for negative data creation
    """

    gbs = [64, 128, 256, 484, 512, 768]
    tbs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if 'gb' in attr:
        type_drive = 'gb'
    else:
        type_drive = 'tb'
    
    orig_amt = int(attr.split(type_drive)[0].strip())
    
    if type_drive == 'gb':
        if orig_amt in gbs:
            gbs.remove(orig_amt)
        return str(random.choice(gbs)) + random.choice([' ', '']) + 'gb'
    
    else:
        if orig_amt in tbs:
            tbs.remove(orig_amt)
        return str(random.choice(tbs)) + random.choice([' ', '']) + 'tb'

def create_neg_laptop_data(df):
    """
    Using the scraped laptop data, create positive pairs
    """
    
    temp = []
    for title in df['title']:
        # Get each major attribute of a laptop
        brand, product_attr, inch, cpu, ram, ssd, hard_drive, other_gb_attrs = get_key_attrs(title)

        # Make sure the product is actually a laptop
        if len(ram) == 0 and len(ssd) == 0 and len(hard_drive) == 0 and len(other_gb_attrs) == 0:
            continue
        
        # Substitute negative attributes
        neg_titles = []
        for x in ram:
            neg_title = title.replace(x, replace_ram_attribute(x))
            neg_titles.append(neg_title)
        
        for x in hard_drive:
            neg_title = title.replace(x, replace_drive_attribute(x))

        for x in ssd:
            neg_title = title.replace(x, replace_drive_attribute(x, True))
            neg_titles.append(neg_title)
        
        if (ram == [] and ssd == []):
            for x in other_gb_attrs:
                neg_title = title.replace(x, replace_other_attribute(x))
                neg_titles.append(neg_title)
        
        MAX_NEG_VARIATIONS = 4
        all_neg_titles_variations = []
        for neg_title in neg_titles:
            temp.append([title, neg_title, 0])

            # Get each major attribute of a laptop
            brand, product_attr, inch, cpu, ram, ssd, hard_drive, other_gb_attrs = get_key_attrs(neg_title)

            # Create a "simple" version of the title using only the major attributes
            shuffle = [cpu, *list(map(lambda x: x.split(' '), ram)),
            *list(map(lambda x: x.split(' '), ssd)), 
            *list(map(lambda x: x.split(' '), hard_drive)), 
            *list(map(lambda x: x.split(' '), other_gb_attrs))]
            random.shuffle(shuffle)

            base_neg_title = brand + product_attr + inch
            for x in shuffle:
                base_neg_title = base_neg_title + x
            
            orig_neg_title = neg_title.split(' ')
            filler_tokens = get_filler_tokens(orig_neg_title, base_neg_title)

            # Generate a list of titles that do not have as many filler words
            new_titles = remove_filler_tokens(orig_neg_title, filler_tokens)

            # Change up the less semantically meaningful attributes on drives/ram
            new_titles = manipulate_title_gbtb(new_titles, ssd, hard_drive, ram)
            all_neg_titles_variations.append(new_titles)

            # Add the negative titles
            for idx, new_title in enumerate(new_titles):
                temp.append([title, new_title, 0])
                if (idx + 1 == MAX_NEG_VARIATIONS):
                    break

        # Pair up titles from the negative title variations
        MAX_NEG_VARIATIONS = 5
        if len(all_neg_titles_variations) > 1:
            pot_combos = list(combinations(range(len(all_neg_titles_variations)), 2))
            for x in range(len(pot_combos)):
                idx_pair = random.choice(pot_combos)
                pot_combos.remove(idx_pair)
                try:   
                    for x in range(MAX_NEG_VARIATIONS):
                        t1 = random.choice(all_neg_titles_variations[idx_pair[0]])
                        t2 = random.choice(all_neg_titles_variations[idx_pair[1]])
                        temp.append([t1, t2, 0])

                except IndexError:
                    pass

                if x + 1 == MAX_NEG_VARIATIONS:
                        break

    return pd.DataFrame(temp, columns=['title_one', 'title_two', 'label'])

def create_retailer_laptop_train_data():
    file_path = 'data/train/retailer_laptop_data.csv'
    
    if not os.path.exists(file_path):
        print('Generating Retailer Laptop train data . . .')
        # Get the laptop data from the different sources
        amazon_laptops = pd.read_csv('data/base/amazon_laptop_titles.csv')
        walmart_laptops = pd.read_csv('data/base/walmart_laptop_titles.csv')
        newegg_laptops = pd.read_csv('data/base/newegg_laptop_titles.csv')

        # Concatenate the data
        laptops = remove_misc(pd.concat([amazon_laptops, walmart_laptops, newegg_laptops]))
        laptops['title'] = laptops['title'].apply(lambda x: remove_stop_words(x, omit_punctuation=['.']))
        laptops = laptops.drop_duplicates(subset=['title'])

        # Create positive titles
        pos_titles = create_pos_laptop_data(laptops)
        pos_titles = pos_titles.drop_duplicates(subset=['title_one', 'title_two'])
        
        # Create negative titles
        neg_titles = create_neg_laptop_data(laptops)
        neg_titles = neg_titles.drop_duplicates(subset=['title_one', 'title_two'])

        # Combine the positive and negative DataFrames and put them in a CSV
        retailer_laptop_df = create_final_data(pos_titles, neg_titles)
        retailer_laptop_df.to_csv(file_path)
    
    else:
        print('Already have Retailer Laptop train data. Moving on . . .')