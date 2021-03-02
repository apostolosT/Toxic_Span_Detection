import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm
import string
import spacy


nlp = spacy.load("en_core_web_sm")



TRAIN_FILE="input/processed/train.csv"
VAL_FILE="input/processed/val.csv"
TEST_FILE="input/processed/test.csv"

RAW_TRAIN_FILE="input/raw/tsd_train.csv"
RAW_VAL_FILE="input/raw/tsd_trial.csv"
RAW_TEST_FILE="input/raw/tsd_test.csv"



def is_whitespace(c):  ##From google-research/bert run_squad.py
    if (
        c == " "
        or c == "\t"
        or c == "\r"
        or c == "\n"
        or ord(c) == 0x202F
        or c in string.whitespace
        or ord(c) == 160
        or ord(c) == 8196
    ):
        return True
    return False


#remove whitespace
def replace_whitespace(text):
    new_text=""
    for c in text:
        if is_whitespace(c):
            new_text+=" "
        else:
            new_text+=c
    return new_text


def make_mappings(text, toxic_spans):
    doc=nlp(text)
    tokens=[]
    offset_mapping=[]
    labels=[]

    for token in doc:
        start,end=token.idx, token.idx + len(token.text)
        
        tokens.append(str(token))

        offset_mapping.append((start,end))
        

        token_span=list(np.arange(start,end))
        
        toxic_span=[i for i in token_span if i in toxic_spans]
        if(toxic_span):

            labels.append(2)
            
        else:
            labels.append(1)
    
    return offset_mapping,labels,tokens 

def convert_to_BIO_tag(labels):
    curr_BIO_tags=[]
    begins=False
    for tag in labels:
        
            if (str(tag)=='2' and begins==False):
                curr_BIO_tags.append("B-TOX")
                begins=True
            
            elif (str(tag)=='2' and begins==True):        
                curr_BIO_tags.append("I-TOX")
            
            elif (str(tag)=='1'):
                curr_BIO_tags.append("O")
                begins=False
        
    return curr_BIO_tags


def prep_data(df):
    
    df.text=[replace_whitespace(text) for text in df.text]

    offset_mapping=[]
    labels=[]
    tokens=[]
    BIO_tags=[]

    for _,row in tqdm(df.iloc[:].iterrows(),total=len(df)):

        offs,lbls,toks=make_mappings(row.text,row.spans)
        offset_mapping.append(offs)
        labels.append(lbls)
        tokens.append(toks)
        BIO_tags.append(convert_to_BIO_tag(lbls))
        del offs,lbls,toks

    df['offset_mapping']=offset_mapping
    df['labels']=labels
    df['tokens']=tokens
    df['BIO_tags']=BIO_tags

    return df


# train_df=pd.read_csv(RAW_TRAIN_FILE)
# train_df.spans=train_df.spans.apply(lambda row:literal_eval(row))
# print("Processing raw train file...")
# train_df=prep_data(train_df)
# train_df.to_csv(TRAIN_FILE,index=False)
# print(f"New processed file created in {TRAIN_FILE}")
# del train_df

# val_df=pd.read_csv(RAW_VAL_FILE)
# val_df.spans=val_df.spans.apply(lambda row:literal_eval(row))
# print("Processing raw val file...")
# val_df=prep_data(val_df)
# val_df.to_csv(VAL_FILE,index=False)
# print(f"New processed file created in {VAL_FILE}")
# del val_df

test_df=pd.read_csv(RAW_TEST_FILE)
test_df.spans=test_df.spans.apply(lambda row:literal_eval(row))
print(f"Processing raw test file {RAW_TEST_FILE}")
test_df=prep_data(test_df)
test_df.to_csv(TEST_FILE,index=False)
print(f"New processed file created in {TEST_FILE}")
del test_df