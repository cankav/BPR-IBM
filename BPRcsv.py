from theano_bpr.utils import load_data_from_csv
from theano_bpr import BPR
import sys

import plotly.plotly as py
import numpy as np

import matplotlib.pyplot as plt

# Loading train data
train_data, users_to_index, items_to_index = load_data_from_csv('WABPRtrainSet.csv')
#print train_data
# Loading test data
test_data, users_to_index, items_to_index = load_data_from_csv('WABPRtestSet.csv', users_to_index, items_to_index)
#print test_data
# Initialising BPR model, 25 latent factors
#in BPR we have number of feature as being 'rank' and is the first argument we pass

bpr = BPR(64, test_data, len(users_to_index.keys()), len(items_to_index.keys()))
#print len(users_to_index)
# Training model, 30 epochs


bpr.train(train_data, epochs=400)
# Testing model
#print bpr.test(test_data)
#print(np.mean(bpr.GM))
#print (type(bpr.GM))

#plt.plot(bpr.GMcount1,'g')
#plt.plot(bpr.GMcount2,'r')
#plt.plot(bpr.GMcount3,'b')
#plt.legend(['Gradient Magnitude<0.5', 'Gradient Magnitude<0.1', 'Gradient Magnitude<0.01'], loc='lower right')
#plt.show()
#plt.plot(bpr.costfunc,'g')
#plt.title('ML-100K dataset')
#plt.xlabel('Training Epoch')
#plt.ylabel('probability')
#plt.show()


plt.plot(bpr.A)
plt.title('WA-Adaptive')
plt.xlabel('Training Epoch')
plt.ylabel('MAP (Mean Average Precision')
plt.show()

#run this in terminal: python BPRcsv.py WABPRtrain.csv WABPRtest.csv
#gedit /home/parisa/.local/lib/python2.7/site-packages/theano_bpr/bpr.py
