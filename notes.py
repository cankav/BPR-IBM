#finding the frequency of items in a list:
    def _find_frequencies(self,y):
		y = list(y)
		counts = collections.Counter(y)
		new_list = sorted(counts, key=counts.get, reverse=True)
		return new_list


def _non_uniform_negativeitem_sampling3(self, n_samples):
    """
    This creates informative negative samples.
    Based on the Adaptive and context dependent sampling
    1.We should sample rank r from geometric distribution
    2.sample factor dimention f from p(f|c)
    3.sort items according to v.,f that is the inverse ranking
    4.Return item j on position r in the sorted list
    """
    q = 0
    if q % (len(train_data) * log(len(train_data))) = 0:
        for f in (1: self._rank):
        sortednegatives[f] = sorted(self.H)  # this gives the ranking of item based on the feature
        mu = numpy.mean(self.H.get_value[f])
        std = numpy.std(self.H[f])
    r = numpy.random.geometric(p=0.35, size=10000)


return r

range(1,self._rank+2):

    def _precompute_ranikng(self,u,f):
	#self.m= self.H.get_value()
	#print(self.m[:,1])#,dir(self.H))
	"""This function is to precompute the ranking for items based on a specific latent feature
	in adaptive sampling"""
	self.score = {}
	self.rank=[]
	self.rankvf={}
	self.user=self.W.get_value()
	self.item=self.H.get_value()
	#self.rank_l = zeros((self._n_users,self._rank)
	for l in list(self._train_items)[0:10]:
		for v in sorted(self._train_dict.keys())[0:10]:
			for f in range(1,self._rank+1):
        			user_feature = self.user[v,f]
				items_feature = self.item[l,f]
       				self.score[v,f,l] = numpy.sign(user_feature)*items_feature
				#at this point I want to sort the scores based on it's values but return the key to my rank (a list) variable
				print self.score
				self.rank = sorted(self.score, key=self.score.get, reverse=True)
				print self.rank
				self.rankvf[(v,f)]=self.rank
	return self.rankvf[(v,f)].append(self.rankvf[(v,f)])





		precision_user = 0.0
        for user in test_dict.keys():
		itemlist = self.top_predictions(user)
		print itemlist
		for l in test_items:
			if l in itemlist:
				precision_user +=1
		    precision.append(numpy.mean(precision_user))



AP_at_n=[]
		AP=[]
		P=0.0
		for r in range(1,100):
			if itemlist[r] in rel_items[user]:
				P += 1
				#print P
			AP.append(P/r)
			#print(AP)
			AP_at_n = numpy.sum(AP)/P
			print AP_at_n
		AP_user.append(AP_at_n)
		#print AP_user

import csv

res = [x, y, z, ....]
csvfile = "<path to output csv or txt>"

#Assuming res is a flat list
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in res:
        writer.writerow([val])