import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import bisect
from tqdm import tqdm

import spacy
import en_core_web_sm
import os

entity_list = {
'Drug',
'Strength',
'Form',
'Dosage',
'Duration',
'Frequency',
'Route',
'ADE',
'Reason',
}

def incorrect_char_only(text):
    return len(text.replace('\n','').replace('\t','').replace(' ','')) == 0

def nextline_only(text):
    return len(text.strip('\n')) == 0

def check_tokens(token, ann_processed):
    result = 'O'
    for item in ann_processed:
        if token[1] >= item[1] and token[1] < item[2]:
            if token[1] == item[1]:
                result = 'B-' + item[0]
            else:
                result = 'I-' + item[0]
            break
    return result

def get_processed_data(ann_path, text_path):

    with open(ann_path,'r') as f:
        ann_lines = f.readlines()
        ann_lines = [item.strip().split('\t') for item in ann_lines if item[0]=='T']
    
    with open(text_path,'r') as f:
        text_lines = f.readlines()
        # text_lines = [item.strip() for item in text_lines]

    text = ''.join(text_lines)
    ann_processed = []
    for line in ann_lines:
        if not line[1].split(' ')[0] in entity_list:
            print('ERROR')
        ann_processed.append([line[1].split(' ')[0], int(line[1].split(' ')[1]), int(line[1].split(' ')[-1]), line[2]])

    doc = nlp(text)
    token_text_list = [token.text for token in doc]
    token_list = [(token.text,token.idx, token.idx+len(token.text)) for token in doc]

    token_label_list = []
    for token in token_list:
        token_label_list.append(check_tokens(token, ann_processed))
    
    df = pd.DataFrame({'tokens':token_text_list, 'labels':token_label_list})
    df_nospace = df[~df['tokens'].apply(incorrect_char_only)]
    return df_nospace

filelist = os.listdir()
nlp = spacy.load("en_core_web_sm")
nlp = en_core_web_sm.load()

os.makedirs('./processed', exist_ok=True)
ann_filelist = [item for item in filelist if '.ann' in item]
txt_filelist = [item.split('.')[0] + '.txt' for item in ann_filelist]

for file_idx in tqdm(range(len(ann_filelist))):
    ann_path = ann_filelist[file_idx]
    text_path = txt_filelist[file_idx]
    print(text_path)
    
    df_nospace = get_processed_data(ann_path, text_path)
    df_nospace.to_csv('./processed/'+text_path.split('.')[0] + '_processed.txt',index=None, header=None, sep='\t')