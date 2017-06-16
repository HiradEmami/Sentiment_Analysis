import  numpy as np
import random
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.sentiment import sentiment_analyzer
from nltk.twitter import Twitter
from nltk.corpus import stopwords, state_union, movie_reviews
from nltk.tokenize import PunktSentenceTokenizer

import re




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


if __name__ =="__main__":
    movieReviews=readMovieReviews()