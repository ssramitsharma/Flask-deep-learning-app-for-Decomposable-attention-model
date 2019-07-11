# Flask-deep-learning-app-for-Decompositional attention model

Recognizing Textual Entailment(RTE) is a task where given a pair of sentences , the system tries to find if the directional relationship between the sentences is that of Entailment, Neutral or Contradiction. The repositiory contains an implementation of  [decomposble attention model](https://github.com/free-variation/spaCy/tree/master/examples/notebooks) with some modification such that the weights are generated.

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

First the model is trained and the weihts are generated. The same weights are then loaded and the mdoel gives prediction .
