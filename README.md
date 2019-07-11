# Flask-deep-learning-app-for-Decompositional attention model

Recognizing Textual Entailment(RTE) is a task where given a pair of sentences , the system tries to find if the directional relationship between the sentences is that of Entailment, Neutral or Contradiction. The repositiory contains an implementation of  [decomposble attention model](https://github.com/free-variation/spaCy/tree/master/examples/notebooks) which is used to generate the weights . The weights thus generated are then used by the flask app to get the prediction.





![alt text](https://github.com/ssramitsharma/Flask-deep-learning-app-for-Decomposable-attention-model/blob/master/flask1.png)


As shown in the figuere there are two boxes in the app , in the first box the text sentece"He is a good boy " is entered and in the sencond box the hypothesis sentence "He is a bad boy"  is entered . 

![alt text](https://github.com/ssramitsharma/Flask-deep-learning-app-for-Decomposable-attention-model/blob/master/flask2.png)

The ouput we get is contradiction as shown in the above figue.
# Requirements:
keras 2.4.2 <br/>
spacy <br/>
flask

# Setting up
pip install keras <br/>
pip install spacy <br/>
python -m spacy download en_vectors_web_lg <br/>
pip install flask


# Initial steps
python atts.py <br/>
python app.py


