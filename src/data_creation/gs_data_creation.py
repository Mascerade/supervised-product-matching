import os
import pandas as pd
import numpy as np
import math
import random
from tqdm import tqdm
from gensim import corpora
from gensim.similarities import SparseMatrixSimilarity
from supervised_product_matching.model_preprocessing import remove_stop_words
from src.common import create_final_data

"""
Much of this algorithm is based on the paper Intermediate Training of BERT for Product Matching
which is by the people who made the WDC Product Corpus:
http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/papers/DI2KG2020_Peeters.pdf
"""

def combinations(total, choose):
    '''
    Simple function to compute combinations
    '''

    return int(math.factorial(total) / (math.factorial(choose) * math.factorial(total - choose)))

def chunk_data():
    '''
    Chunks the WDC Product Corpus into files of 100,000 rows each
    '''

    chunk_size = 100000
    batch = 1
    for chunk in pd.read_json('data/base/offers_corpus_english_v2.json.gz', lines=True, nrows= 100000000000000, chunksize=chunk_size):
        chunk.to_json('data/base/product_corpus/chunk' + str(batch) + '.json')
        batch += 1

def generate_computer_data():
    '''
    Gets the computer data from the WDC Product Corpus
    '''
    chunk_size = 100000
    computer_df = pd.DataFrame()
    for chunk in pd.read_json('data/base/offers_corpus_english_v2.json.gz', lines=True, nrows= 100000000000000, chunksize=chunk_size):
        computer_df = computer_df.append(chunk[chunk['category'].values == 'Computers_and_Accessories'])
    return computer_df

def extract_key_features(cluster):
    '''
    Simplies the DataFrames extracted from the WDC Product Corpus
    Only includes the ID, description, title, and title + description
    '''

    new_cluster = cluster.loc[:, ("id", "description", "title")]
    new_cluster["title"] = new_cluster["title"].map(lambda x: remove_stop_words(x))
    new_cluster["description"] = new_cluster["description"].map(lambda x: remove_stop_words(str(x)))
    new_cluster["titleDesc"] = new_cluster["title"].map(lambda x: x.split(" ")) + new_cluster["description"].map(lambda x: x.split(" ")).map(lambda x: x[0:6])
    return new_cluster

def get_valid_clusters(df):
    '''
    Returns the IDs of all the clusters that have more than 1 but less than 80 titles in them
    '''

    MAX_CLUSTER_SIZE = 80
    valid_clusters = (((df['cluster_id'].value_counts() > 1) & 
                        (df['cluster_id'].value_counts() <= MAX_CLUSTER_SIZE)))

    valid_clusters = list(valid_clusters[valid_clusters == True].index)
    all_clusters = df[df['cluster_id'].isin(valid_clusters)]['cluster_id'].values
    return set(all_clusters)

def create_pos_from_cluster(data, cluster_id):
    '''
    Creates positive pairs from a cluster
    '''

    MAX_PAIRS = 16
    cluster = data.loc[data["cluster_id"].values == cluster_id]
    cluster = extract_key_features(cluster)
    max_combos = combinations(len(cluster), 2)
    
    dictionary = corpora.Dictionary(cluster["titleDesc"])
    cluster_dict = [dictionary.doc2bow(title) for title in cluster["title"].map(lambda x: x.split(" "))]
    sim_matrix = np.array(SparseMatrixSimilarity(cluster_dict, num_features=len(dictionary)))
    
    # Because the matrix is redundant (the rows and columns represent the same titles)
    # we set the bottom half of the similarities (including the diagonal) to 100
    # so that we don't have to worry about them when doing argmin()
    for row in range(sim_matrix.shape[0]):
        for column in range(sim_matrix.shape[1]):
            if (row >= column):
                sim_matrix[row][column] = 100
    
    # If the maximum amount of combinations we can make is less than our set max,
    # set the maximum to the max combos
    if max_combos < MAX_PAIRS:
        MAX_PAIRS = max_combos
    
    # Half of the pairs should be hard positives and the other half random
    hard_pos = MAX_PAIRS // 2
    random_pos = MAX_PAIRS - hard_pos
    
    pairs = []

    # Hard positives are those that are from the same cluster, but with the least similarity
    for x in range(hard_pos):
        # Keep getting the pairs with the lowest similarity score
        min_sim = np.unravel_index(sim_matrix.argmin(), sim_matrix.shape)
        pair = [cluster["title"].iloc[min_sim[0]], cluster["title"].iloc[min_sim[1]], 1]
        pairs.append(pair)
        sim_matrix[min_sim[0]][min_sim[1]] = 100
    
    # The amount of available pairs (given that some are gone from hard positive creation)
    avail_indices = np.argwhere(sim_matrix != 100)

    # Get random pairs within the same cluster
    for x in range(random_pos):
        ran_idx = random.sample(list(range(len(avail_indices))), 1)
        choice = avail_indices[ran_idx][0]
        pair = [cluster["title"].iloc[choice[0]],
                cluster["title"].iloc[choice[1]], 1]
        pairs.append(pair)
        avail_indices = np.delete(avail_indices, ran_idx, 0)
    
    return pd.DataFrame(pairs, columns=["title_one", "title_two", "label"])

def create_neg_from_cluster(data, cluster_id, all_clusters):
    '''
    Creates negative pairs from a cluster
    '''

    # Get the cluster
    cluster = data.loc[data["cluster_id"].values == cluster_id]
    cluster = extract_key_features(cluster)
    pairs = []
    hard_neg = len(cluster) // 2
    
    # Hard negatives are those that are from different clusters, but we get the pair with the highest similarity
    for row in range(hard_neg):
        # Keep choosing random titles until we get one that is not our own
        neg_cluster_id = cluster_id        
        while neg_cluster_id == cluster_id:
            neg_cluster_id = random.choice(all_clusters)
        
        # Extract data about this cluster
        neg_cluster = data.loc[data["cluster_id"].values == neg_cluster_id].copy()
        neg_cluster = extract_key_features(neg_cluster)
        
        # Add the current title of the cluster to the beginning of this random cluster so that
        # the first row in the similarity matrix will refer to this title
        neg_cluster = pd.concat([pd.DataFrame([cluster.iloc[row].values], columns=["id", "description", "title", "titleDesc"]),
                                 neg_cluster])
        
        # Get the similarity between the title and the random cluster
        dictionary = corpora.Dictionary(neg_cluster["titleDesc"])
        neg_cluster_dict = [dictionary.doc2bow(title) for title in neg_cluster["title"].map(lambda x: x.split(" "))]
        sim_matrix = np.array(SparseMatrixSimilarity(neg_cluster_dict, num_features=len(dictionary)))
        
        # First row is the similarity between the current title and the rest of the random cluster
        # so get the max similarity of this (+1 is because we don't include the similarity with ourself)
        max_val = sim_matrix[0][1:].argmax() + 1
        
        # Add the pair
        pair = [cluster["title"].iloc[row], neg_cluster["title"].iloc[max_val], 0]
        pairs.append(pair)
    
    for row in range(hard_neg, len(cluster)):
        # Keep choosing random titles until we get one that is not our own
        neg_cluster_id = cluster_id
        while neg_cluster_id == cluster_id:
            neg_cluster_id = random.choice(all_clusters)
        
        # Randomly get a title from the random cluster
        neg_cluster = data.loc[data["cluster_id"].values == neg_cluster_id].copy()
        neg_cluster = extract_key_features(neg_cluster)
        neg_title = neg_cluster["title"].iloc[random.choice(list(range(len(neg_cluster))))]
        
        # Add the pair
        pair = [cluster["title"].iloc[row], neg_title, 0]
        pairs.append(pair)
    
    return pd.DataFrame(pairs, columns=["title_one", "title_two", "label"])

def create_computer_gs_data():
    file_path = 'data/train/wdc_computers.csv'
    if not os.path.exists(file_path):
        print('Generating Gold Standard Computer data (takes a long time) . . .')
        # Get the titles from the WDC Product Corpus
        if not os.path.exists('data/base/computer_wdc_whole_no_duplicates.csv'):
            computer_df = generate_computer_data()
            computer_df = computer_df.drop_duplicates('title')
            computer_df.to_csv('data/base/computer_wdc_whole_no_duplicates.csv')
        
        else:
            computer_df = pd.read_csv('data/base/computer_wdc_whole_no_duplicates.csv')
        
        # Get "good" clusters from the data
        valid_clusters = list(get_valid_clusters(computer_df))
        computer_train_wdc_pos = pd.DataFrame(columns=["title_one", "title_two", "label"])
        computer_train_wdc_neg = pd.DataFrame(columns=["title_one", "title_two", "label"])

        # Positive data creation
        print('    Generating postive example . . .')
        for cluster in tqdm(valid_clusters):
            computer_train_wdc_pos = computer_train_wdc_pos.append(create_pos_from_cluster(computer_df, cluster))

        # Negative data creation
        print('    Generating negative examples . . .')
        for cluster in tqdm(valid_clusters):
            computer_train_wdc_neg = computer_train_wdc_neg.append(create_neg_from_cluster(computer_df, cluster, valid_clusters))

        # Concatenate the data
        computer_train_wdc = create_final_data(computer_train_wdc_pos, computer_train_wdc_neg)
        computer_train_wdc.to_csv('data/train/wdc_computers.csv')
    
    else:
        print('Already have Gold Standard Computer Data. Moving on . . .')