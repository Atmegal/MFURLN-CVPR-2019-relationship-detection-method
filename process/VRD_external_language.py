import csv
import pandas as pd
import scipy.io as sci
import numpy as np
# load GloVe embedding.
glove = pd.read_csv('./process/database/glove.6B.300d.txt', sep=' ',quoting=csv.QUOTE_NONE, header=None)
glove.set_index(0, inplace=True)
# load vocabulary.
objectList = sci.loadmat('./data/objectListN.mat')
Olist = objectList['objectListN']

relationList = sci.loadmat('./data/relationListN.mat')
rList = relationList['relationListN']

vocab = rList[0]
embedding = np.zeros([len(vocab), len(glove.columns)], np.float64)
not_found = []
for i in range(len(vocab)):
    word = vocab[i][0]
    if word in glove.index:
        embedding[i] = glove.loc[word]
    else:
	if word == 'next to':
		embedding[i] = glove.loc['next']
	elif word == 'sleep next to':
		embedding[i] = (glove.loc['next'] + glove.loc['sleep'])/2.0
	elif word == 'sit next to':
		embedding[i] = (glove.loc['next'] + glove.loc['sit'])/2.0
	elif word == 'stand next to':
		embedding[i] = (glove.loc['next'] + glove.loc['stand'])/2.0
	elif word == 'park next':
		embedding[i] = (glove.loc['next'] + glove.loc['park'])/2.0
	elif word == 'walk next to':
		embedding[i] = (glove.loc['next'] + glove.loc['walk'])/2.0
	elif word == 'stand behind':
		embedding[i] = (glove.loc['behind'] + glove.loc['stand'])/2.0
	elif word == 'sit behind':
		embedding[i] = (glove.loc['behind'] + glove.loc['sit'])/2.0
	elif word == 'park behind':
		embedding[i] = (glove.loc['behind'] + glove.loc['park'])/2.0
	elif word == 'in the front of':
		embedding[i] = glove.loc['front'] 
	elif word == 'stand under':
		embedding[i] = (glove.loc['under'] + glove.loc['stand'])/2.0
 	elif word == 'sit under':
		embedding[i] = (glove.loc['under'] + glove.loc['sit'])/2.0
	elif word == 'stand under':
		embedding[i] = (glove.loc['under'] + glove.loc['stand'])/2.0
	elif word == 'walk to':
		embedding[i] = (glove.loc['walk'] + glove.loc['to'])/2.0
	elif word == 'walk past':
		embedding[i] = (glove.loc['walk'] + glove.loc['past'])/2.0
	elif word == 'walk beside':
		embedding[i] = (glove.loc['walk'] + glove.loc['beside'])/2.0
	elif word == 'on the top of':
		embedding[i] = glove.loc['top'] 
	elif word == 'on the left of':
		embedding[i] = glove.loc['left'] 
	elif word == 'on the right of':
		embedding[i] = glove.loc['right'] 
	elif word == 'sit on':
		embedding[i] =  (glove.loc['on'] + glove.loc['sit'])/2.0
	elif word == 'stand on':
		embedding[i] =  (glove.loc['on'] + glove.loc['stand'])/2.0
	elif word == 'attach to':
		embedding[i] =  (glove.loc['to'] + glove.loc['attach'])/2.0
	elif word == 'adjacent to':
		embedding[i] =  (glove.loc['to'] + glove.loc['adjacent'])/2.0
	elif word == 'drive on':
		embedding[i] =  (glove.loc['on'] + glove.loc['drive'])/2.0
 	elif word == 'taller than':
		embedding[i] =  (glove.loc['taller'] + glove.loc['than'])/2.0
	elif word == 'park on':
		embedding[i] =  (glove.loc['on'] + glove.loc['park'])/2.0
	elif word == 'lying on':
		embedding[i] =  (glove.loc['lying'] + glove.loc['on'])/2.0
	elif word == 'lean on':
		embedding[i] =  (glove.loc['on'] + glove.loc['lean'])/2.0
 	elif word == 'play with':
		embedding[i] =  (glove.loc['play'] + glove.loc['with'])/2.0
	elif word == 'sleep on':
		embedding[i] =  (glove.loc['sleep'] + glove.loc['on'])/2.0
	elif word == 'outside of':
		embedding[i] =  (glove.loc['outside'] + glove.loc['of'])/2.0
 	elif word == 'rest on':
		embedding[i] =  (glove.loc['rest'] + glove.loc['on'])/2.0 
  	elif word == 'skate on':
		embedding[i] =  (glove.loc['skate'] + glove.loc['on'])/2.0 
	else:
		not_found.append(i)
		print(i,embedding)
print('Not found:\n', vocab[not_found])
# For words not included in GloVe, set to average of found embeddings.
embedding_avg = np.mean(embedding, 0)
embedding[not_found] += embedding_avg
# For the first word 'UNK', set its embedding to 0
#embedding[0] = 0

np.save('./input/VRD/rList_word_embedding.npy',
        embedding.astype(np.float32))

#vocab = pd.read_csv('data/ActivityNet-QA-preprocess/vocab.txt', header=None)[0]
vocab = Olist[0]
embedding = np.zeros([len(vocab), len(glove.columns)], np.float64)
not_found = []
for i in range(len(vocab)):
    word = vocab[i][0]
    if word in glove.index:
        embedding[i] = glove.loc[word]
    else:
	if word == 'traffic light':
		embedding[i] = (glove.loc['traffic'] + glove.loc['light'])/2.0
	elif word == 'trash can':
		embedding[i] = (glove.loc['trash'] + glove.loc['can'])/2.0
	else:
        	embedding[i][i] = 1
		not_found.append(i)
print('Not found:\n', vocab[not_found])
# For words not included in GloVe, set to average of found embeddings.
embedding_avg = np.mean(embedding, 0)
embedding[not_found] += embedding_avg
# For the first word 'UNK', set its embedding to 0
#embedding[0] = 0

np.save('./input/VRD/oList_word_embedding.npy',
        embedding.astype(np.float32))
