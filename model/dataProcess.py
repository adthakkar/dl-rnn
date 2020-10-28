#!/usr/bin/python3 

import argparse
import os
import numpy as np
import re
import random

from pprint import pprint
from string import punctuation

def parse_jhu_dataset(fname, label):
    review_list = []

    with open(fname, 'r') as f:
        reviews = f.read().split('</review>')

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
            rev = parse_jhu_dataset(name[0], name[1])
            reviews = reviews + rev
    
    random.shuffle(reviews)
    return reviews

def load_train_data():
    review_list = []
    train_text = []
    train_label = []
     
    review_list = load_jhu_dataset() 
    
    for review in review_list:
        rev_text = review['review_text'].lower()
        rev_punct_removed = ''.join([c for c in rev_text if c not in punctuation])
         
        train_text.append(rev_punct_removed)
        train_label.append(review['label'])

    print(train_text)   

    return 

def main():
    load_train_data()

if __name__ == '__main__':
    main()
