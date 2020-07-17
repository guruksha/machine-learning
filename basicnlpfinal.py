#basic nlp operations and pipelining
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

messages = pd.read_csv("SMSSpamCollection", sep="\t", names = ['label','message'])

messages['length'] =messages['message'].apply(len)

import string 

from nltk.corpus import stopwords
def text_process(mess):
	nonpunc = [c for c in mess if c not in string.punctuation]
	nonpunc  = "".join(nonpunc)
	clean_mess  =  [word for word in nonpunc.split() if word.lower() not in stopwords.words("english")]
	return clean_mess
from sklearn.feature_extraction.text import CountVectorizer

bow_tranformer = CountVectorizer(analyzer =text_process).fit(messages['message'])

# mesg4 = messages['message'][3]
# bw4 = bow_tranformer.transform([mesg4])

message_bow = bow_tranformer.transform(messages['message'])
print(message_bow.nnz)

from sklearn.feature_extraction.text import TfidfTransformer
tfifd_trasfomer = TfidfTransformer().fit(message_bow)
#tfifd4 = tfifd_trasfomer.transform(bw4)

messages_tfidf = tfifd_trasfomer.transform(message_bow)

from sklearn.naive_bayes import MultinomialNB

sdm = MultinomialNB().fit(messages_tfidf, messages['label'])
#print(sdm.predict(tfifd4))


from sklearn.model_selection import train_test_split
m_train, m_test, l_train, l_test  = train_test_split(messages['message'], messages['label'], test_size = .2)

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
	('bow', CountVectorizer(analyzer = text_process)),
	('tftid', TfidfTransformer()),
	('classifier', MultinomialNB())
	])

pipeline.fit(m_train, l_train)
predictions = pipeline.predict(m_test)

from sklearn.metrics import classification_report

print(classification_report(predictions,l_test))
