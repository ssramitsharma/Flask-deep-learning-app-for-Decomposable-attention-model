from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
import keras
from keras.layers import LSTM,Embedding,Dropout,Activation,Dense,Bidirectional,GRU
from keras.models import Sequential,Input,Model
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import  pad_sequences
import time
import spacy
import numpy as np
import time
import json
#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
#sys.path.append(os.path.abspath("./model"))

#initalize our flask app
#app = Flask(__name__)
#global vars for easy reusability

#initialize these variables

from flask import Flask,render_template,url_for,request
#import pandas as pd 
import pickle

import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
import keras
from keras.layers import LSTM,Embedding,Dropout,Activation,Dense,Bidirectional,GRU
from keras.models import Sequential,Input,Model
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import  pad_sequences
import time
import spacy
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt 
import time
import json
#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
#sys.path.append(os.path.abspath("./model"))

#initalize our flask app
#app = Flask(__name__)
#global vars for easy reusability

#initialize these variables

import numpy as np
glove_filename = 'glove.6B.50d.txt'

import tensorflow as tf
import numpy as np
import keras.models
from keras.models import model_from_json

import tensorflow as tf


#import ujson as json
from keras.utils import to_categorical


nlp = spacy.load('en_vectors_web_lg')            
            
            
def inits(): 
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load woeights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    #loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    graph = tf.get_default_graph()

    return loaded_model,graph
model, graph = inits()



import spacy
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt 
import time
import json
import pandas as pd

def create_dataset(nlp, texts, hypotheses, num_oov, max_length, norm_vectors = True):
    sents = texts + hypotheses
    
    # the extra +1 is for a zero vector represting NULL for padding
    num_vectors = max(lex.rank for lex in nlp.vocab) + 2 
    
    # create random vectors for OOV tokens
    oov = np.random.normal(size=(num_oov, nlp.vocab.vectors_length))
    oov = oov / oov.sum(axis=1, keepdims=True)
    
    vectors = np.zeros((num_vectors + num_oov, nlp.vocab.vectors_length), dtype='float32')
    vectors[num_vectors:, ] = oov
    for lex in nlp.vocab:
        if lex.has_vector and lex.vector_norm > 0:
            vectors[lex.rank + 1] = lex.vector / lex.vector_norm if norm_vectors == True else lex.vector
            
    sents_as_ids = []
    for sent in sents:
        doc = nlp(sent)
        word_ids = []
        
        for i, token in enumerate(doc):
            # skip odd spaces from tokenizer
            if token.has_vector and token.vector_norm == 0:
                continue
                
            if i > max_length:
                break
                
            if token.has_vector:
                word_ids.append(token.rank + 1)
            else:
                # if we don't have a vector, pick an OOV entry
                word_ids.append(token.rank % num_oov + num_vectors) 
                
        # there must be a simpler way of generating padded arrays from lists...
        word_id_vec = np.zeros((max_length), dtype='int')
        clipped_len = min(max_length, len(word_ids))
        word_id_vec[:clipped_len] = word_ids[:clipped_len]
        sents_as_ids.append(word_id_vec)
        
        
    return vectors, np.array(sents_as_ids[:len(texts)]), np.array(sents_as_ids[len(texts):])


# In[10]:


#sem_vectors, text_vectors, hypothesis_vectors = create_dataset(nlp, texts, hypotheses, 100, 50, True)



app = Flask(__name__)
            
            
            




@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST','SPOST'])
def predict():
    k = 0

    if request.method == 'POST':
        text = request.form['message']
        text = str(text)
        train_sentence1 = [text]
        hypothesis = request.form['messages']
        hypothesis = str(hypothesis)
        hypothesis = [hypothesis]
   
        sem_vectors, text_vectors, hypothesis_vectors = create_dataset(nlp, train_sentence1, hypothesis, 100, 50, True)
        

        #global graph
        with graph.as_default():
            y = model.predict([text_vectors, hypothesis_vectors],verbose = 0)
            print(y,'y')
            n_values = 3
            t = None
            c = np.eye(n_values, dtype=int)[np.argmax(y, axis=1)]
            print(c,'c')
            
            if str(c) == "[[1 0 0]]":
                t = 0

            if str(c) == "[[0 1 0]]":
                t = 1
            if str(c) == "[[0 0 1]]":
                t = 2

   
           
    
        
        
    
    

        return render_template('result.html',prediction = t)



if __name__ == '__main__':
	app.run(debug=True)
