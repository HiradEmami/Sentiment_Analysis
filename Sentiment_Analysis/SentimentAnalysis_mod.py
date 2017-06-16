import  numpy as np
import random
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.sentiment import sentiment_analyzer
from nltk.twitter import Twitter
from nltk.corpus import stopwords, state_union, movie_reviews
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from nltk.classify.scikitlearn import SklearnClassifier
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
    print (stop_words)
    wordlist= dat
    filtered= [w for w in wordlist if not w in stop_words]
    return filtered

def filterSymbols(dat):
    dic = ["!", "?",
           ".", "@",
           "[", "]",
           "{", "}",
           "*", "&",
           "(", ")",
           "_","-",
           ",",'"',
           "'","...",
           "--","`",
           "$","#",
           "%","^","+","-","/",":",";"
           ]
    wordlist = dat
    filtered = [w for w in wordlist if not w in dic]
    return filtered



#reading Movie Reviews that were creadted and assign it to an array
#the first item is the actual sentiment / polarity and the second item is the review
def readMovieReviews():
    print ('Reading Movie Reviews:')
    readMe = open('MovieReviews.txt', 'r').readlines()

    # Creates a list containing 5 lists, each of 8 items, all set to 0
    w, h = 2, len(readMe)
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

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def convertToDocument():
    w, h = 2, len(movieReviews)
    doc = [[0 for x in range(w)] for y in range(h)]
    for i in range(0, len(movieReviews)):
        cat=movieReviews[i][0]
        tokenized= wordTokenize(movieReviews[i][1])
        setTokenized= set(tokenized)
        doc[i][1]=cat
        doc[i][0]=setTokenized

    return doc

def saveClassifier():
    save_classifier = open("naivebayes.pickle", "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


if __name__ =="__main__":
    #reading the corpus
    movieReviews=readMovieReviews()
    #init the POS Tag
    custom_Sent_Tagg = train_Punktokenizer()
    sentence_Example = "Hello Lovely World. This is Just to test Our POS Tagger!"
    #identifyNamedEntity(sentence_Example)

    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words=filterStopWords(all_words)
    all_words=filterSymbols(all_words)

    all_words=nltk.FreqDist(all_words)

    word_features = [w[0] for w in sorted(all_words.items(), key=lambda (k, v): v, reverse=True)[:5000]]

    #print (word_features)
   # print ('daster ')
   # print((movie_reviews.words('neg/cv000_29416.txt')))
   # print ('daster ')
   # k=wordTokenize(movieReviews[100][1])
   # print k
   # print ('daster ')
   # print set(k)
    #print ('daster')
   # print (find_features(k))

    documents=convertToDocument()

    featuresets = [(  find_features(rev), category) for (rev,category) in documents]

    # set that we'll train our classifier with
    training_set = featuresets[:1900]

    # set that we'll test against.
    testing_set = featuresets[1900:]

    classifier = nltk.NaiveBayesClassifier.train(training_set)

    print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

    classifier.show_most_informative_features(15)

    #if you want to save it
   # saveClassifier()