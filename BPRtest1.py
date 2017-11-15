from theano_bpr.utils import load_data_from_csv
from theano_bpr import BPR
import sys

if len(sys.argv) != 3:
    print "Usage: ./example.py training_data.csv testing_data.csv"
sys.exit(1)

# Loading train data
train_data, users_to_index, items_to_index = load_data_from_csv(sys.argv[1])
print train_data[0,5]
# Loading test data
test_data, users_to_index, items_to_index = load_data_from_csv(sys.argv[2], users_to_index, items_to_index)
print test_data[0:5]
# Initialising BPR model, 10 latent factors
bpr = BPR(10, len(users_to_index.keys()), len(items_to_index.keys()))

# Training model, 30 epochs
bpr.train(train_data, epochs=30)
# Testing model
print bpr.test(test_data)

