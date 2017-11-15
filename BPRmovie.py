from theano_bpr.utils import load_data_from_movielens
training_data, users_to_index, items_to_index = load_data_from_movielens('http://files.grouplens.org/datasets/movielens/ml-100k/ub.base', 3)
print training_data

testing_data, users_to_index, items_to_index = load_data_from_movielens('http://files.grouplens.org/datasets/movielens/ml-100k/ub.test', 3, users_to_index, items_to_index)


print users_to_index
from theano_bpr import BPR

bpr = BPR(10, len(users_to_index.keys()), len(items_to_index.keys()))

bpr.train(training_data, epochs=50)
bpr.test(testing_data)