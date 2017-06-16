import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.sentiment import sentiment_analyzer
from nltk.twitter import Twitter
from nltk.corpus import stopwords, state_union
from nltk.tokenize import PunktSentenceTokenizer
import re






#varriables

tw = Twitter()

#tw.tweets(follow=['759251'],keywords='iran', limit=20) #sample from the public stream

def sentenceTokenize(dat):
    return sent_tokenize(dat,'english')

def wordTokenize(dat):
    return word_tokenize(dat,'english')

#start process_tweet
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

def fetchOfflineTwits(argKeyword, argNumber, argStream):
    return tw.tweets(keywords=argKeyword, stream=argStream,
                     to_screen=False,limit=argNumber)

def filterStopWords(dat):
    stop_words =set(stopwords.words("english"))
    wordlist= wordTokenize(dat)
    filtered= [w for w in wordlist if not w in stop_words]
    return filtered

def posTagSentence(dat):
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




if __name__ =="__main__":
    # nltk.download()

    #sentence_Example = "I cant begin to express, how much I hate computational Discourse. Why cant I just finish Lab4?"
    sentence_Example="Hello Lovely cat. You are amazing, funny and smart"
    print (sentence_Example)

    print ('sentence tokenized')
    for i in sentenceTokenize(sentence_Example):
        print (i)
    print('!done!')

    print('Word Tokenized:')
    for i in wordTokenize(sentence_Example):
        print (i)

    print ('&&&&&&&&&&&&&&&&&')
    print filterStopWords(sentence_Example)

    training_Text_Punk = state_union.raw("2006-GWBush.txt")
    training_Text_Punk2 = state_union.raw("2005-GWBush.txt")
    custom_Sent_Tagg = PunktSentenceTokenizer(training_Text_Punk)
    custom_Sent_Tagg =PunktSentenceTokenizer(training_Text_Punk2)


    for i in posTagSentence(sentence_Example):
        print i
        for j in i:
            if j[1]=='JJ':
                print j
                print j[1]
    #twitts =fetchOfflineTwits('#globalwarming',1,False )

    #print processTweet(twitts)
