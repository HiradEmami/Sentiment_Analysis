import  numpy as np
import random
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.sentiment import sentiment_analyzer
from nltk.twitter import Twitter
from nltk.corpus import stopwords, state_union, movie_reviews
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer

import re


#function to tokenize a text based on its Sentences (split sentences)
#Returns a List of tokenized sentences
def sentenceTokenize(dat):
    return sent_tokenize(dat,'english')

#function to tokenize a sentence based on its words (split words)
#Returns a List of tokenized words
def wordTokenize(dat):
    return word_tokenize(dat,'english')

#function to tokenize a sentence based on its words and remove all the stop words
#Returns a List of tokenized words that do not include stop words
def filterStopWords(dat):
    stop_words =set(stopwords.words("english"))
    wordlist= wordTokenize(dat)
    filtered= [w for w in wordlist if not w in stop_words]
    return filtered


#reading Movie Reviews that were creadted and assign it to an array
#the first item is the actual sentiment / polarity and the second item is the review
def readMovieReviews():
    print ('Reading Movie Reviews:')
    readMe = open('MovieReviews.txt', 'r').readlines()

    # Creates a list containing 5 lists, each of 8 items, all set to 0
    w, h = 2, len(readMe);
    document = [[0 for x in range(w)] for y in range(h)]

    for i in range(0, len(readMe)):
        temp1= readMe[i].split("] <")
        temp2= temp1[1].split(" >, ")
        polarity=temp2[0]
        review=temp2[1]
        document[i][0]=polarity
        document[i][1]=review
    print ('Reviews are Imported')
    print ('Length of Corpus: '+str(len(document)))
    print ('example Review:')
    print ("polarity: "+document[1000][0])
    print ("Review: "+document[1000][1])
    return document

def posTagSentence(dat):
   Tokenized= custom_Sent_Tagg.tokenize(dat)
   AnswerList=[]
   try:
       for i in Tokenized:
           wordList=wordTokenize(i)
           tagged=nltk.pos_tag(wordList)
           AnswerList.append(tagged)
   except Exception as e:
       print (str(e))
   return AnswerList

def train_Punktokenizer():
    training_Text_Punk = state_union.raw("2006-GWBush.txt")
    training_Text_Punk2 = state_union.raw("2005-GWBush.txt")
    custom_Sent_Tagg = PunktSentenceTokenizer(training_Text_Punk)
    custom_Sent_Tagg =PunktSentenceTokenizer(training_Text_Punk2)

    print ('Training Custom Tokenizer Completed')
    return custom_Sent_Tagg

def identifyNamedEntity(dat):
   Tokenized= custom_Sent_Tagg.tokenize(dat)
   AnswerList=[]
   try:
       for i in Tokenized:
           wordList=wordTokenize(i)
           tagged=nltk.pos_tag(wordList)

           namedENT= nltk.ne_chunk(tagged,binary=True)

           print (len(namedENT))
           print namedENT
           AnswerList.append(tagged)
   except Exception as e:
       print (str(e))
   return AnswerList

def lemmatizeThisWord(dat):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(dat)

if __name__ =="__main__":
    #reading the corpus
    movieReviews=readMovieReviews()
    #init the POS Tag
    custom_Sent_Tagg = train_Punktokenizer()
    sentence_Example = "Hello Lovely World. This is Just to test Our POS Tagger!"
    #identifyNamedEntity(sentence_Example)