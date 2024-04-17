
# Loading the required libraries
# Standard Libraries
import os
import numpy as np 
import pandas as pd
import random
import keras
import torch
import tensorflow as tf
import plotly
import optuna
from optuna import Trial
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import log_loss
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn import metrics

import re
import string
import json
import emoji
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Setting adjustments for the notebook ----------------------------------------------
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

# Custom Functions -----------------------------------------------------------------
SEED = 99
def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(SEED)
random_seed(SEED)

# Installing new packages -----------------------------------------------------------
# !pip install sentence_transformers
# !pip install optuna
# !pip install lxml
# !pip install wordcloud





# Custom Functionalities ###############################################################################################
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
                       "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am",
                       "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", 
                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                       "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 
                       "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 
                       "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                       "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                       "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                       "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 
                       "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                       "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", 
                       "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", 
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have",
                       "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'u.s':'america', 'e.g':'for example',
                       "em":"them","u":"you", "homie":"friend", "cus":"because", "give a shit":"care", "jeez": "jesus", "UK":"United Kingdom", 
                       "won't've": "will not have", "Thnx": "thanks","Yeeeaaaah":"yeah", "rip":"rest in peace","ENTP":"extrovert", "INTP": "introvert","da":"the",
                       "yeeey": "yes", "pics":"pictures", "Sheed":"shit","ik":"i know","doggo":"dog","fn":"fucking", "irl":"in real life","y i k e s":"yikes", 
                       "S A T I R E": "satire", "blitz'd" :"blitzed", "fella":"fellow","smh":"shaking my head","boi" :"boy"}

punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                 "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 
                 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}

mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
                'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 
                'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 
                'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 
                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                'demonetisation': 'demonetization', "gatekepping":"gate keeping",'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 
                'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', "wilk":"walk", 'fcuk': 'fuck', 'Etherium': 'Ethereum',
                "ananarchist":" an anarchist","veggie":"vegetable", 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 
                'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 
                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 
                'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'narcissit': 'narcissist', 'bigdata': 'big data', 
                '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 
                'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization',
                "accudentally":"accidentally","heff":" have", "bawlin":"bawling", "lawd":"lord","comin":"coming"}

def clean_text(text):
    '''Clean emoji, Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = emoji.demojize(text)
    text = re.sub(r'\:(.*?)\:','',text)
    text = str(text).lower()    #Making Text Lowercase
    text = re.sub('\[.*?\]', '', text)
    #The next 2 lines remove html text
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text

def clean_contractions(text, mapping):
    '''Clean contraction using contraction mapping'''    
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    for word in mapping.keys():
        if ""+word+"" in text:
            text = text.replace(""+word+"", ""+mapping[word]+"")
    #Remove Punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text

def clean_special_chars(text, punct, mapping):
    '''Cleans special characters present(if any)'''   
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

def correct_spelling(x, dic):
    '''Corrects common spelling errors'''   
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

def remove_space(text):
    '''Removes awkward spaces'''   
    #Removes awkward spaces 
    text = text.strip()
    text = text.split()
    return " ".join(text)

def text_preprocessing_pipeline(text):
    '''Cleaning and parsing the text.'''
    text = clean_text(text)
    text = clean_contractions(text, contraction_mapping)
    text = clean_special_chars(text, punct, punct_mapping)
    text = correct_spelling(text, mispell_dict)
    text = remove_space(text)
    return text

# Other Custom Functionalities ===========================================================================================
import tensorflow_hub as hub
import tensorflow_text as text

# preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
# https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3
# encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1")
# https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1
# preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1")
# preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
# encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1")

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def get_embeddings(sentences):
    '''return BERT-like embeddings of input text
    Args:
    - sentences: list of strings
    Output:
    - BERT-like embeddings: tf.Tensor of shape=(len(sentences), 768)
    '''
    preprocessed_text = bert_preprocess(sentences)
    # return encoder(preprocessed_text)['pooled_output']
    df = pd.DataFrame(bert_encoder(preprocessed_text)['pooled_output'])
    df.index = sentences.index
    return df

def get_embeddings_demo(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']

def get_labels_testset(ydataset:pd.DataFrame, threshold=0.35):
    def get_labels(predicted_list, thresh=threshold):
        mlb =[(i1,c1)for i1, c1 in enumerate(y_train.columns)]    
        temp_list = sorted([(i,c) for i,c in enumerate(list(predicted_list))],key = lambda x: x[1], reverse=True)
        tag_list = [item1 for item1 in temp_list if item1[1]>=thresh] # here 0.35 is the threshold i choose
        tags = [item[1] for item2 in tag_list[:5] for item in mlb if item2[0] == item[0] ] # here I choose to get top 5 labels only if there are more than that
        return tags
    
    ydataset = pd.DataFrame(ydataset)
    ydataset.columns = ydataset.columns
    return ydataset.apply(lambda x: get_labels(x,threshold), axis=1)

def relabel_preds(ypreds):
    y_pred=[]
    for sample in  ypreds:
        y_pred.append([1 if i>=0.5 else 0 for i in sample ] )
    return np.array(y_pred)

def train_val_test_split(X,Y,tt_ratio=0.2,tv_ratio=0.2, seed=999):
    xtrain_tmp, x_test, y_train_tmp, y_test = train_test_split(X, Y, test_size=tt_ratio, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(xtrain_tmp, y_train_tmp, test_size=tv_ratio, random_state=seed)
    return {'x_train':x_train, 'y_train':y_train,
            'x_val':x_val, 'y_val':y_val,
            'x_test':x_test, 'y_test':y_test}

def EmotionMapping(emotion_list):
    map_list = []
    
    for i in emotion_list:
        if i in ekman_mapping['anger']:
            map_list.append('anger')
        if i in ekman_mapping['disgust']:
            map_list.append('disgust')
        if i in ekman_mapping['fear']:
            map_list.append('fear')
        if i in ekman_mapping['joy']:
            map_list.append('joy')
        if i in ekman_mapping['sadness']:
            map_list.append('sadness')
        if i in ekman_mapping['surprise']:
            map_list.append('surprise')
        if i == 'neutral':
            map_list.append('neutral')
            
    return map_list

ekman_mapping = dict({"anger": ["anger", "annoyance", "disapproval"],
                      "disgust": ["disgust"],
                      "fear": ["fear", "nervousness"],
                      "joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism",
                              "relief", "pride", "admiration", "desire", "caring"],
                      "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
                      "surprise": ["surprise", "realization", "confusion", "curiosity"],
                      "neutral": ['neutral']})

def return_values(emotion_list):
    anger=0
    disgust=0
    fear=0
    joy=0
    sadness=0
    surprise=0
    neutral=0

    for i in emotion_list:
        if i=='anger':
            anger=1
        elif i=='disgust':
            disgust=1
        elif i=='fear':
            fear=1
        elif i=='joy':
            joy=1
        elif i=='sadness':
            sadness=1
        elif i=='surprise':
            surprise=1
        elif i=='neutral':
            neutral=1
    return [anger, disgust, fear, joy, sadness, surprise, neutral]

def shorten_emotions(x):
    emotions=list(x[x==1].index)
    shortened_emotions=EmotionMapping(emotions)
    return return_values(shortened_emotions)

# Auto_Construct Best Model
def auto_construct(best_params:dict, n_inputs, n_classes):
    from keras.models import Sequential
    best_model = Sequential()
    for i in range(0,best_params['n_layers']):
        key_dict = {key: value for (key, value) in best_params.items() if str(i) in key}
        if(i==0):
            best_model.add(Dense(key_dict['n_units_l'+str(i)], input_dim=n_inputs, kernel_initializer=key_dict['kernel_initializer'+str(i)], 
                                 activation=key_dict['activation'+str(i)]))
            best_model.add(Dropout(key_dict['dropout'+str(i)]))
        else:
            best_model.add(Dense(key_dict['n_units_l'+str(i)], kernel_initializer=key_dict['kernel_initializer'+str(i)], 
                                 activation=key_dict['activation'+str(i)]))
            best_model.add(Dropout(key_dict['dropout'+str(i)]))
    best_model.add(Dense(n_classes, activation=tf.keras.activations.sigmoid)) #output Layer
    return best_model

# Auto Construct BERT Model from best params
def auto_construct_bert(best_params:dict, n_classes):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)    
    link = tf.keras.layers.Dropout(0.1)(outputs['pooled_output'])
    for i in range(0,best_params['n_layers']):
        key_dict = {key: value for (key, value) in best_params.items() if str(i) in key}
        link = tf.keras.layers.Dense(key_dict['n_units_l'+str(i)], kernel_initializer=key_dict['kernel_initializer'+str(i)], 
                                     activation=key_dict['activation'+str(i)])(link)
        link = tf.keras.layers.Dropout(key_dict['dropout'+str(i)])(link)
    op_layer = tf.keras.layers.Dense(n_classes, activation='sigmoid', name="output")(link) #output Layer
        
    # Defining the BERT Model Blueprint
    return tf.keras.Model(inputs=[text_input], outputs = [op_layer])

# Multi-Label classification metrics ############################################################################
def emr(y_true, y_pred):
    n = len(y_true)
    row_indicators = np.all(y_true == y_pred, axis = 1) # axis = 1 will check for equality along rows.
    exact_match_count = np.sum(row_indicators)
    return exact_match_count/n

# Print the required metrics
def metrics(ytrain_true, ytrain_pred, ytest_true, ytest_pred, target_names, val_test_mode='val'):
    from sklearn.metrics import classification_report
    
    print("Train Metrics #########################################################","\n")
    print("Train EMR (Accuracy) :", emr(y_true=ytrain_true, y_pred=ytrain_pred),"\n")
    print(classification_report(y_true=ytrain_true, y_pred=ytrain_pred, output_dict=False, target_names=target_names))
    print("\n")
    
    if val_test_mode=='val':
        set_name='Validation'
    elif(val_test_mode=='Test'):
        set_name='Test'
        
    print(set_name," Metrics #####################################################","\n")
    print(set_name, "EMR(Accuracy) :", emr(y_true=ytest_true, y_pred=ytest_pred),"\n")
    print(classification_report(y_true=ytest_true, y_pred=ytest_pred, output_dict=False, target_names=target_names))
    
def best_threshold_val(y_true, y_pred_probs, verbose=0):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    thresholds = np.arange(0.1, 1.0, 0.01)
    acc_dict=dict({})

    # Evaluate the model for each threshold value
    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        prec = precision_score(y_true=y_true, y_pred=y_pred, average='samples')
        rec = recall_score(y_true=y_true, y_pred=y_pred, average='samples')
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
        acc_dict[threshold]=acc
        # roc_auc = roc_auc_score(y_true=y_train, y_pred=bestann_model_train_predictions, average='samples')
        if verbose==1:
            print(f"Threshold={threshold}: Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    return round(max(acc_dict, key=acc_dict.get), 2)

def extract_emotions_from_probs(x):
    cols=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
    try:
        index_of_1 = x.index(1)
        return cols[index_of_1]
    except:
        return 'No emotion detected!'
    
def emotion_prediction_demo(text, model, threshold):
    for sent in text:
        preproc_text = text_preprocessing_pipeline(sent)
        embeds = get_embeddings_demo([preproc_text])
        probs = model.predict(embeds, verbose=0)
        predictions = (probs >= threshold).astype(int)
        print(sent, "->", extract_emotions_from_probs(predictions))