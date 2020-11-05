#!/usr/bin/python3 

import argparse
import os
import numpy as np
import re
import random
import pandas as pd
import matplotlib.pyplot as plt
 
from pprint import pprint
from string import punctuation
from collections import Counter


def parse_csv_dataset(fname):
    reviews = []
    review_dict = {} 

    print("parse_csv_dataset(): Parsing data from %s" % (fname))

    data = pd.read_csv(fname, usecols=['Score', 'Text'])
    for rev in data.itertuples():
        review_dict['rating'] = float(rev[1])
        review_dict['review_text'] = rev[2]
        reviews.append(review_dict)

    return reviews


def parse_jhu_dataset(fname, label):
    review_list = []

    with open(fname, 'r') as f:
        reviews = f.read().split('</review>')

        print("parse_jhu_dataset(): Parsing data from %s" % (fname))

        for review in (reviews):
            review_dict = {} 

            try:
                review_dict['title'] = review.split("<product_name>")[1].split("</product_name>")[0].strip() 
                review_dict['product_type'] = review.split("<product_type>")[1].split("</product_type>")[0].strip()
                review_dict['rating'] = review.split("<rating>")[1].split("</rating>")[0].strip()
                review_dict['review_text'] = review.split("<review_text>")[1].split("</review_text>")[0].strip()
                review_dict['label'] = label
                review_list.append(review_dict)
            except IndexError:
                print("Error - IndexError occurred while parsing [%s]. Skipping review number" % (fname))
                pass

    return review_list
            

def load_jhu_dataset():
    reviews = []

    with open('/home/ec2-user/dl-rnn/dataset/jhu_dataset', 'r') as f:
        filenames = f.read().splitlines()
        
        for fname in filenames:
            name = fname.split('%')
            print("load_jhu_dataset(): Loading and parsing data from following file %s" % (name))

            rev = parse_jhu_dataset(name[0], name[1])
            reviews = reviews + rev
    
    random.shuffle(reviews)
    return reviews

def load_csv_dataset():
    reviews = []

    with open('/home/ec2-user/dl-rnn/dataset/food_reviews', 'r') as f:
        filenames = f.read().splitlines()

        for fname in filenames:
            print("load_csv_dataset(): Loading and parsing data from following file %s" % (fname))
            rev = parse_csv_dataset(fname)
            reviews = reviews + rev

    random.shuffle(reviews)
    

    return reviews

def tokenize_text(text_list):
    text_join = ' '.join(text_list)
    words = text_join.split()
    count = Counter(words)
    
    total = len(words)
    common_words = count.most_common(total) 

    words_to_int = {w:i+1 for i, (w,c) in enumerate(common_words)}

    return words_to_int

def encode_reviews(text_list, vocab):
    encoded_reviews = []

    for rev in text_list:
        tmp = [vocab[w] for w in rev.split()] 
        encoded_reviews.append(tmp)

    return encoded_reviews

def plot_data(rev_len, plot_name = 'xyz'):
    pd.Series(rev_len).hist()
    plt.show()
    plt.savefig(plot_name)
    
    s = pd.Series(rev_len).describe([0.25, 0.5, 0.75, 0.99])
    print(s)

def plot_review_data(rev_list, data_type):
    rev_text = []
    for rev in rev_list:
        rev_text.append(rev['review_text'].strip())

    rev_len = [len(w.split()) for w in rev_text]
    plot_data(rev_len, data_type)


def load_train_data():
    review_list = []
    train_text = []
    train_label = []
    
    rev1 = load_jhu_dataset()
    rev2 = load_csv_dataset() 
   
    plot_review_data(rev1, 'jhu_data.png')
    plot_review_data(rev2, 'food_review_data.png')
     
    review_list = rev1 + rev2 
    
    for review in review_list:
        rev_text = review['review_text'].lower().strip()
        rev_punct_removed = ''.join([c for c in rev_text if c not in punctuation])
         
        train_text.append(rev_punct_removed)
        train_label.append(review['rating'])
 
    vocabulary = tokenize_text(train_text)
    reviews_encoded = encode_reviews(train_text, vocabulary)

    rev_len = [len(w) for w in reviews_encoded]
    plot_data(rev_len, 'tokenized_review_len.png')

    return 

def main():
    #load_csv_dataset() 
    load_train_data()

if __name__ == '__main__':
    main()
