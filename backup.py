# theano-bpr
#
# Copyright (c) 2014 British Broadcasting Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import theano, numpy, math
import theano.tensor as T
import time
import sys
from collections import defaultdict
import collections

print 'hi Parisa'


class BPR(object):
    def __init__(self, rank, mytest, n_users, n_items, lambda_u=0.0025, lambda_i=0.0025, lambda_j=0.00025,
                 lambda_bias=0.0, learning_rate=0.05):
        """
          Creates a new object for training and testing a Bayesian
          Personalised Ranking (BPR) Matrix Factorisation
          model, as described by Rendle et al. in:

            http://arxiv.org/abs/1205.2618

          This model tries to predict a ranking of items for each user
          from a viewing history.
          It's also used in a variety of other use-cases, such
          as matrix completion, link prediction and tag recommendation.

          `rank` is the number of latent features in the matrix
          factorisation model.

          `n_users` is the number of users and `n_items` is the
          number of items.

          The regularisation parameters can be overridden using
          `lambda_u`, `lambda_i` and `lambda_j`. They correspond
          to each three types of updates.

          The learning rate can be overridden using `learning_rate`.

          This object uses the Theano library for training the model, meaning
          it can run on a GPU through CUDA. To make sure your Theano
          install is using the GPU, see:

            http://deeplearning.net/software/theano/tutorial/using_gpu.html

          When running on CPU, we recommend using OpenBLAS.

            http://www.openblas.net/

          Example use (10 latent dimensions, 100 users, 50 items) for
          training:

          >>> from theano_bpr import BPR
          >>> bpr = BPR(10, 100, 50)
          >>> from numpy.random import randint
          >>> train_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.train(train_data)

          This object also has a method for testing, which will return
          the Area Under Curve for a test set.

          >>> test_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.test(test_data)

          (This should give an AUC of around 0.5 as the training and
          testing set are chosen at random)
        """
        self._rank = rank
        self._n_users = n_users
        self._n_items = n_items
        self._lambda_u = lambda_u
        self._lambda_i = lambda_i
        self._lambda_j = lambda_j
        self._lambda_bias = lambda_bias
        self._learning_rate = learning_rate
        self._train_users = set()
        self._train_items = set()
        self._train_dict = {}
        self._configure_theano()
        self._generate_train_model_function()
        self.GM = []
        self.GMcount1 = []
        self.GMcount2 = []
        self.GMcount3 = []
        self.randuser_sampling = numpy.zeros(1)
        self.y = []
        self.costfunc = []
        self.m = {}
        self.user = {}
        self.item = {}
        self._mytest = mytest

    def _configure_theano(self):
        """
          Configures Theano to run in fast mode
          and using 32-bit floats.
        """
        theano.config.mode = 'FAST_RUN'
        theano.config.floatX = 'float32'

    def _count1(self, x):
        """
          These three are counters for Gradient Magnitudes that are less than a value.
        """
        counter = sum(i < 0.5 for i in x)
        return counter

    def _count2(self, x):
        counter = sum(i < 0.1 for i in x)
        return counter

    def _count3(self, x):
        counter = sum(i < 0.01 for i in x)
        return counter

    def _precompute_ranikng(self):

        """This function is to precompute the ranking for items based on a specific latent feature
        in adaptive sampling"""
        self.user_feature = []
        self.score = {}  # adictionary to save the rating predictions
        self.rank = []  # list of sorted items
        self.itemlist = {}  # list of sorted items for each list
        self.user = self.W.get_value()
        self.item = self.H.get_value()
        self.features = range(1, self._rank)
        for v, f in zip(list(self._train_users), self.features):
            self.user_feature = self.user[v, f]
            # print user_feature
            for l in list(self._train_items):
                items_feature = self.item[l, f]
                # print ('--'+ str(items_feature))
                self.score[l] = numpy.sign(user_feature) * items_feature
                # at this point I want to sort the scores based on it's values but return the key to my rank (a list) variable
                # print self.score
                self.rank = sorted(self.score, key=self.score.get, reverse=True)
            # print self.rank[1:10]
            self.itemlist[(v, f)] = self.rank
        return self.itemlist[(v, f)]

    # we are not using this but the author has in on his psuedo code
    def _precompute_mean(self, user):
        self.mu = {}
        self.std = {}
        self.user = self.W.get_value()
        self.item = self.H.get_value()
        self.features = range(1, self._rank)
        for v, f in zip(list(self._train_users), self.features):
            user_feature = self.user[v, f]
            self.std = numpy.std(self.item[:, f])
            dist = numpy.sign(user_feature) * self.std[f]
        return self.std[(user, f)]

    def _generate_train_model_function(self):
        """
          Generates the train model function in Theano.
          This is a straight port of the objective function
          described in the BPR paper.

          We want to learn a matrix factorisation

            U = W.H^T

          where U is the user-item matrix, W is a user-factor
          matrix and H is an item-factor matrix, so that
          it maximises the difference between
          W[u,:].H[i,:]^T and W[u,:].H[j,:]^T,
          where `i` is a positive item
          (one the user `u` has watched) and `j` a negative item
          (one the user `u` hasn't watched).
        """
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')

        self.W = theano.shared(numpy.random.random((self._n_users, self._rank)).astype('float32'), name='W')
        self.H = theano.shared(numpy.random.random((self._n_items, self._rank)).astype('float32'), name='H')

        self.user = self.W.get_value()
        self.item = self.H.get_value()

        self.B = theano.shared(numpy.zeros(self._n_items).astype('float32'), name='B')

        x_ui = T.dot(self.W[u], self.H[i].T).diagonal()
        x_uj = T.dot(self.W[u], self.H[j].T).diagonal()

        x_uij = self.B[i] - self.B[j] + x_ui - x_uj

        obj = T.sum(T.log(T.nnet.sigmoid(x_uij)) - self._lambda_u * (self.W[u] ** 2).sum(axis=1) - self._lambda_i * (
        self.H[i] ** 2).sum(axis=1) - self._lambda_j * (self.H[j] ** 2).sum(axis=1) - self._lambda_bias * (
                    self.B[i] ** 2 + self.B[j] ** 2))
        cost = - obj
        # print(x_uij.eval())
        g_cost_W = T.grad(cost=cost, wrt=self.W)
        g_cost_H = T.grad(cost=cost, wrt=self.H)
        g_cost_B = T.grad(cost=cost, wrt=self.B)
        self.get_g_cost_B = theano.function(inputs=[u, i, j], outputs=g_cost_B)

        updates = [(self.W, self.W - self._learning_rate * g_cost_W), (self.H, self.H - self._learning_rate * g_cost_H),
                   (self.B, self.B - self._learning_rate * g_cost_B)]

        self.train_model = theano.function(inputs=[u, i, j], outputs=[cost, T.nnet.sigmoid(x_uij)], updates=updates)

    def train(self, train_data, epochs=30, batch_size=10000):
        """
          Trains the BPR Matrix Factorisation model using Stochastic
          Gradient Descent and minibatches over `train_data`.

          `train_data` is an array of (user_index, item_index) tuples.

          We first create a set of random samples from `train_data` for
          training, of size `epochs` * size of `train_data`.

          We then iterate through the resulting training samples by
          batches of length `batch_size`, and run one iteration of gradient
          descent for the batch.
        """

        self.A = []
        if len(train_data) < batch_size:
            sys.stderr.write(
                "WARNING: Batch size is greater than number of training samples, switching to a batch size of %s\n" % str(
                    len(train_data)))
            batch_size = len(train_data)

        self._train_dict, self._train_users, self._train_items, self.sorteditems = self._data_to_dict(train_data)
        # The next 4 lines initializes the item list for the first iteration when we still did not run the precompute
        self.ranklist = {}
        keys = itertools.product(list(self._train_users), range(1, self._rank))
        a = list(self._train_items)
        self.ranklist = dict.fromkeys(keys, a)
        n_sgd_samples = len(train_data) * epochs
        sgd_users, sgd_pos_items = self._uniform_user_sampling(n_sgd_samples)
        sgd_neg_items = self._non_uniform_negativeitem_sampling1(
            n_sgd_samples)  # this I changed for negative samples to be different
        z = 0
        t2 = t1 = t0 = time.time()
        q = 0
        while (z + 1) * batch_size < n_sgd_samples:
            [cost, grad_mag] = self.train_model(
                sgd_users[z * batch_size: (z + 1) * batch_size],
                sgd_pos_items[z * batch_size: (z + 1) * batch_size],
                sgd_neg_items[z * batch_size: (z + 1) * batch_size]
            )
            # We pre-compute the ranking every couple of learning iterations. The self.ranklist is then used for sampling3
            while q % ((n_sgd_samples) * math.log(n_sgd_samples)) == 0:
                self.ranklist = self._precompute_ranikng()
                q += 1
                print self.ranklist

            self.MAP = self.test(self._mytest)
            self.A.append(self.MAP)
            # print self.A
            sys.stderr.write(str(
                cost) + ' -- ')  # str(1-grad_mag)+' -- '+ str(z)+' -- '+ str(n_sgd_samples))#+' -- '+ str(batch_size) +' -- '+ str(len(self.GM))) #+' \n'+ str(x_ui[0:5]) + '\n ' + str(x_uj[0:5] )+ '\n')
            self.GM.append(1 - grad_mag)
            GMcounter1 = self._count1(1 - grad_mag) / float(
                batch_size)  # should be divided by batch size as we have this much observations in one epoch#for WA is 3183
            self.GMcount1.append(GMcounter1)
            # print(('---'+str(self.GMcount1)))
            GMcounter2 = self._count2(1 - grad_mag) / float(batch_size)
            self.GMcount2.append(GMcounter2)
            GMcounter3 = self._count3(1 - grad_mag) / float(batch_size)
            self.GMcount3.append(GMcounter3)
            # print str(self.sorteditems[0:20])
            self.costfunc.append(cost)
            # print str(self.costfunc[0:20])

            self._writecsv(self.A, "/home/parisa/PycharmProjects/replication/movie_1.csv")
            z += 1
            t2 = time.time()
            sys.stderr.write("\rProcessed %s ( %.2f%% ) in %.4f seconds" % (
            str(z * batch_size), 100.0 * float(z * batch_size) / n_sgd_samples, t2 - t1))
            sys.stderr.flush()
            t1 = t2
            if n_sgd_samples > 0:
                sys.stderr.write(
                    "\nTotal training time %.2f seconds; %e per sample\n" % (t2 - t0, (t2 - t0) / n_sgd_samples))
                sys.stderr.flush()

    def _non_uniform_negativeitem_sampling1(self, n_samples):

        """
        This creats negative sample for the original BPR
        In fact This is a UNIFORM negative item sampling
        """
        sgd_users1 = self.randuser_sampling
        sgd_neg_items = []
        for sgd_user in sgd_users1:
            neg_item = numpy.random.randint(self._n_items)
            while neg_item in self._train_dict[sgd_user]:
                neg_item = numpy.random.randint(self._n_items)
            sgd_neg_items.append(neg_item)

        return sgd_neg_items

    def _non_uniform_negativeitem_sampling2(self, n_samples):

        """
        This creates informative negative samples.
        Based on the static and global sampling
        First I should find the most frequent items in my dataset
        Then if the current user did not use this popular item this is an informative negative sample
        """
        sgd_users1 = self.randuser_sampling
        sgd_neg_items = []

        print self.sorteditems[0:10]
        for sgd_user in sgd_users1:
            i = 0
            neg_item = self.sorteditems[i]
            while neg_item in self._train_dict[sgd_user]:
                i += 1
                neg_item = self.sorteditems[i]
            sgd_neg_items.append(neg_item)
        # print sgd_neg_items
        return sgd_neg_items

    def _non_uniform_negativeitem_sampling3(self, n_samples):
        """
        This creates informative negative samples.
        Based on the Adaptive and context dependent sampling
        1.We should sample rank r from geometric distribution
        2.sample factor dimention f from p(f|c)
        3.sort items according to v.,f that is the inverse ranking
        4.Return item j on position r in the sorted list
        """

        r = numpy.int(numpy.random.exponential(scale=10, size=None))
        sgd_users1 = self.randuser_sampling
        sgd_neg_items = []
        for sgd_user in sgd_users1:
            a = range(1, self._rank)
            f = numpy.random.choice(a, size=None, replace=True, p=None)
            ranking = self.ranklist[(sgd_user, f)]
            if numpy.sign(self.user_feature) > 0:
                neg_item = ranking[r]
                else:
                neg_item = ranking[-r + 1]
            sgd_neg_items.append(neg_item)

    return sgd_neg_items


def _uniform_user_sampling(self, n_samples):
    """
      Creates `n_samples` random samples from training data for performing Stochastic
      Gradient Descent. We start by uniformly sampling users,
      and then sample a positive and a negative item for each
      user sample.
    """
    sys.stderr.write("Generating %s random training samples\n" % str(n_samples))
    sgd_users = numpy.array(list(self._train_users))[numpy.random.randint(len(list(self._train_users)), size=n_samples)]
    self.randuser_sampling = sgd_users  # this is added to be used in my negative item sampling I want same users for negative samples
    sgd_pos_items = []  # this is also changed
    for sgd_user in sgd_users:
        pos_item = self._train_dict[sgd_user][numpy.random.randint(len(self._train_dict[sgd_user]))]
        sgd_pos_items.append(pos_item)
        # neg_item = numpy.random.randint(self._n_items)
        # while neg_item in self._train_dict[sgd_user]:
        #  neg_item = numpy.random.randint(self._n_items)
        # sgd_neg_items.append(neg_item)
    return sgd_users, sgd_pos_items  # , sgd_neg_items


def predictions(self, user_index):
    """
      Computes item predictions for `user_index`.
      Returns an array of prediction values for each item
      in the dataset.
    """
    w = self.W.get_value()
    h = self.H.get_value()
    b = self.B.get_value()
    user_vector = w[user_index, :]
    return user_vector.dot(h.T) + b


def prediction(self, user_index, item_index):
    """
      Predicts the preference of a given `user_index`
      for a gven `item_index`.
    """
    return self.predictions(user_index)[item_index]


def top_predictions(self, user_index, topn=1000):
    """
      Returns the item indices of the top predictions
      for `user_index`. The number of predictions to return
      can be set via `topn`.
      This won't return any of the items associated with `user_index`
      in the training set.
    """

    return [
               item_index for item_index in numpy.argsort(self.predictions(user_index))
               if item_index not in self._train_dict[user_index]
           ][::-1][:topn]


def test(self, test_data):
    """
      Computes the Area Under Curve (AUC) on `test_data`.

      `test_data` is an array of (user_index, item_index) tuples.

      During this computation we ignore users and items
      that didn't appear in the training data, to allow
      for non-overlapping training and testing sets.
    """
    test_dict, test_users, test_items, test_sorteditems = self._data_to_dict(test_data)
    auc_values = []
    z = 0
    AP_user = []
    itemlist = {}
    rel_items = {}

    for user in range(1, self._n_users):
        rel_items[user] = test_dict[user] + (self._train_dict[user])
        predictions = self.predictions(user)
        itemlist[user] = self.top_predictions(user)
        # AUC calculation
        auc_for_user = 0.0
        n = 0
        for pos_item in test_dict[user]:
            if pos_item in self._train_items:
                for neg_item in self._train_items:
                    if neg_item not in test_dict[user] and neg_item not in self._train_dict[user]:
                        n += 1
                        if predictions[pos_item] > predictions[neg_item]:
                            auc_for_user += 1

        if n > 0:
            auc_for_user /= n
            auc_values.append(auc_for_user)

        z += 1
        if z % 100 == 0 and len(auc_values) > 0:
            sys.stderr.write("\rCurrent AUC mean (%s samples): %0.5f" % (str(z), numpy.mean(auc_values)))
            sys.stderr.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()

    MAP = self._MAP(list(rel_items.values()), list(itemlist.values()))
    print MAP

    return MAP  # numpy.mean(auc_values)#,


def _writecsv(self, value, path):
    import csv

    res = list(value)
    csvfile = path

    # Assuming res is a flat list
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in res:
            writer.writerow([val])


def _AP(self, actual, predicted, k=1000):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
         A list of elements that are to be predicted (order doesn't matter)
    predicted : list
            A list of predicted elements (order does matter)
    k : int, optional
    The maximum number of predicted elements
    Returns
    -------
    score : double
        The average precision at k over the input lists
    """
    # if len(predicted)>k:
    #	predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(
            predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def _MAP(self, actual, predicted, k=1000):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return numpy.mean([self._AP(a, p, k) for a, p in zip(actual, predicted)])


def _data_to_dict(self, data):
    data_dict = defaultdict(list)
    items = set()
    item_freq = {}
    item = data[1]
    for (user, item) in data:
        if item not in item_freq:
            item_freq[item] = 1
        else:
            item_freq[item] += 1
    sorteditems = sorted(item_freq, key=item_freq.get, reverse=True)

    for (user, item) in data:
        data_dict[user].append(item)
        items.add(item)
    return data_dict, set(data_dict.keys()), items, sorteditems