import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from gtts import gTTS
import os
from django.http.response import JsonResponse
warnings.filterwarnings('ignore')
# Imports the Google Cloud client library
from google.cloud import speech

# Instantiates a client

import speech_recognition as sr
import nltk
from nltk.stem import WordNetLemmatizer
from django.http import HttpResponse
import datetime
from django.template import RequestContext, Template, Context
from django.template.loader import get_template, render_to_string
#for downloading package files can be commented after First run
nltk.download('popular', quiet=True)
nltk.download('nps_chat', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')

posts = nltk.corpus.nps_chat.xml_posts()[:10000]



    
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# Recognised input types
# Greet
# "Bye"/>
# "Clarify"/>
# "Continuer"/>
# "Emotion"/>
# "Emphasis"/>
# "Greet"/>
# "Reject"/>
# "Statement"/>
# "System"/>
# "nAnswer"/>
# "whQuestion"/>
# "yAnswer"/>
# "ynQuestion"/>
# "Other"


# colour palet
def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


def prYellow(skk): print("\033[93m {}\033[00m".format(skk))


def prLightPurple(skk): print("\033[94m {}\033[00m".format(skk))


def prPurple(skk): print("\033[95m {}\033[00m".format(skk))


def prCyan(skk): print("\033[96m {}\033[00m".format(skk))


def prLightGray(skk): print("\033[97m {}\033[00m".format(skk))


def prBlack(skk): print("\033[98m {}\033[00m".format(skk))


# Reading in the input_corpus
with open('base.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenisation
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

ABOUT_INPUTS = ("About Delphito", "About Delphito's Creator", "About Artificial Intelligence")
ABOUT_RESPONSES = ["Hello, my name is Delphito and I'm your Personalized Artificial Intelligence. I'm developed by Enrique Frese Arroyo."]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def about(sentence):
    """If user's input is an about question, return a response"""
    for word in sentence.split():
        if word.lower() in ABOUT_INPUTS:
            return (ABOUT_RESPONSES)


# Generating response and processing
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response



        
        #  pip install numpy, sklearn, gtts, speech_recognition
        # pip install  PyAudio
        # pip install  nltk
        # python -m pip install Django

def start(request):
    #start_template(request)
    html = render_to_string('main/main.html', using=None)
    # html = "<html><body>It is now %s.</body></html>" % now
    return HttpResponse(html)

   
def other(request):
    # To Recognise input type as QUES.
    client = speech.SpeechClient()
    # The name of the audio file to transcribe
    gcs_uri = "gs://cloud-samples-data/speech/brooklyn_bridge.raw"
    
    audio = speech.RecognitionAudio(uri=gcs_uri)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        )
    # Detects speech in the audio file
    response = client.recognize(config=config, audio=audio)
    
    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))


def start_function(request):
   
    file = "file.mp3"
    flag = True
    fst = "Hello, my name is Delphito and I'm your Personalized Artificial Intelligence. I'm developed by Enrique Frese Arroyo."
    tts = gTTS(text=fst, lang="en")
    tts.save(file)
    os.system("mpg123 " + file)
    r = sr.Recognizer()
    prYellow(fst)
    # Taking voice input and processing
    while (flag == True):
        with sr.Microphone() as source:
            audio = r.listen(source)
        try:
            user_response = format(r.recognize_google(audio))
            print("\033[91m {}\033[00m".format("YOU SAID : " + user_response))
        except sr.UnknownValueError:
            prYellow("Oops! Didn't catch that")
            pass
        # user_response = input()
        # user_response = user_response.lower()
    
        clas = classifier.classify(dialogue_act_features(user_response))
        if (clas != 'Bye'):
            if (clas == 'Emotion'):
                flag = False
                prYellow("Delphi: You are welcome..")
            else:
                if (greeting(user_response) != None):
                    print("\033[93m {}\033[00m".format("Delphi: " + greeting(user_response)))
                else:
                    print("\033[93m {}\033[00m".format("Delphi: ", end=""))
                    res = (response(user_response))
                    prYellow(res)
                    sent_tokens.remove(user_response)
                    tts = gTTS(res, 'com')
                    tts.save(file)
                    os.system("mpg123 " + file)
        else:
            flag = False
            prYellow("Delphi: Bye! take care..")
            
    