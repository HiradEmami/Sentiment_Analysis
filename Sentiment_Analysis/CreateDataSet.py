

import  numpy as np
import random
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.sentiment import sentiment_analyzer
from nltk.twitter import Twitter
from nltk.corpus import stopwords, state_union, movie_reviews
from nltk.tokenize import PunktSentenceTokenizer

import re



def WORKONTHIS():
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    print(all_words.most_common(15))
    print(all_words["stupid"])

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

#function too draw a chunk based on sentence, text
def drawChunk(dat):
   Tokenized= custom_Sent_Tagg.tokenize(dat)
   AnswerList=[]
   try:
       for i in Tokenized:
           wordList=wordTokenize(i)
           tagged=nltk.pos_tag(wordList)
           AnswerList.append(tagged)

           chunkGram= r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
           chunkParser = nltk.RegexpParser(chunkGram)
           chunked = chunkParser.parse(tagged)
           print chunked
           chunked.draw()
   except Exception as e:
       print (str(e))
   return AnswerList

#Function that uses our Custom POS tagger
#Retursn POS tagged words in the sentence as a list
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

def convertReviewToSentence(sentence):
    converted =""
    for i in range(0,len(sentence[0])):
        if i>0:
            if sentence[0][i]=="'":
                converted += (""+ sentence[0][i])
            else:
                if sentence[0][i - 1] == "'":
                    converted += (""+ sentence[0][i])
                else:
                    converted += (" "+ sentence[0][i])
        else:
            converted += (sentence[0][i])
    return converted

def identifyReviewPolarity(sentence):
    converted =""
    if sentence[1]=="neg":
        converted="negative"
    else:
        converted="positive"
    return converted

def createSymbolDic():
    dic=["!","?",
         ".", "@",
         "[", "]",
         "{", "}",
         "*", "&",
         "(", ")",
         ]
    return dic

def saveMovieReview():
    saveFile = open('MovieReviews.txt', 'w')
    for i in range(0,len(documents)):
        review=convertReviewToSentence(documents[i])
        polarity=identifyReviewPolarity(documents[i])
        saveFile.write("\n["+str(i)+"] <"+polarity+" >, "+review)
    print ('Movie Reviews Saved!')
    saveFile.close()
if __name__=="__main__":

    print("Initializing Symbol Dictionary:")
    symobol_dic=createSymbolDic()
    print("Symbol Dictionary Created")

    print('Initializing Corpus (please wait):')
    # gathering The corpus
    try:

        documents = [(list(movie_reviews.words(fileid)), category)
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]
        print ('Corpus Successfully Imported')
        print('length of document :')
        print len(documents)
        print('Movie Review Categories:', movie_reviews.categories())

        print('Example Movie Review: ')
        print(documents[1][0])
        print ('converted Sentence:')
        print convertReviewToSentence(documents[1])
        print ('Polarity: ',identifyReviewPolarity(documents[1]))

        print ('Shuffeling Movie Reviews:')
        random.shuffle(documents)
        print ('Completed!')

        print ('Saving Movie Reviews:')
        saveMovieReview()



    except Exception as e:
        print (str(e))

    print ('Initializing and Training Custom_POS_Sentence_Tagger:')
    custom_Sent_Tagg=train_Punktokenizer()
    sentence_Example = "Hello Lovely World. This is Just to test Our POS Tagger!"
    print ('Testing POS Tagger:')
    print ('Example Sentence: ',sentence_Example)
    print ('POS Tagged: ', posTagSentence(sentence_Example))






