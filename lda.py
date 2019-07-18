import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import logsumexp

class LDA(object):

	# Initializes with the number of topics, and other parmeters for class to run
	def __init__(self, num_topics, text_data, words, num_words, processed=False, max_iter=50, tol=100):
	self.tol = tol
	self.num_topics = num_topics
	self.words = words
	self.max_iter = max_iter
	self.num_words = num_words

        # Process data if needed
	if processed == True:
		self.vocab = text_data
		self.num_docs = text_data.shape[1]
	else:
		self.text_data = text_data
		self.num_docs = int(text_data[-1][0])
		self.gen_vocab_matrix()
	self.theta = np.random.dirichlet(2**np.random.randint(0, 4, size=self.num_topics))
	self.beta = np.random.dirichlet(2**np.random.randint(0, 4, size=self.num_topics))

        # Initilize random params
	for n in xrange(1, self.num_words):
		self.beta = np.vstack((self.beta, np.random.dirichlet(2**np.random.randint(0, 4, size=self.num_topics))))

	def gen_vocab_matrix(self):
		self.vocab = np.zeros((self.num_words, self.num_docs))
		for row in self.text_data:
			doc_idx = int(row[0]) - 1
			word_idx = int(row[1]) - 1
			self.vocab[word_idx, doc_idx] = row[2]

	def loglike(self, m='ch'):
		return ((np.dot(self.vocab.T, np.log(self.beta)) + np.log(self.theta)) * self.qzd()).sum()

	def M_step(self):
		# Uses posterior estimates to compute theta
		self.theta = self.qzd().sum(axis=0)
		self.theta = self.theta / self.theta.sum()

		# Use estimates and words to compute betas
		beta_num = np.dot(self.vocab, self.qzd())
		self.beta = beta_num / (self.qzd().T * self.vocab.sum(0)).T.sum(0)
		self.beta[self.beta == 0] = 10 ** -10

	def E_step(self):
                # Estimate 
		log_gamma_num = np.dot(self.vocab.T, np.log(self.beta)) + np.log(self.theta)
		log_gamma_den = logsumexp(log_gamma_num, axis = 1)
		self.log_gamma = (log_gamma_num.T - log_gamma_den).T

	def qzd(self):
		return np.exp(self.log_gamma)

	def print_topics(self, n_words=5):
		with open(self.words, 'r') as f:
			word_dict_lines = f.readlines()
		for row in np.array([i.split(' ')[1].replace('\n', '') for i in word_dict_lines])[self.beta.T.argsort(axis=1)[:, -n_words:][:, ::-1]].tolist():
			print ', '.join(row)
			print ''

	def fit(self):
		self.loglikes = [0]
		for i in xrange(self.max_iter):
			self.E_step()
			self.M_step()
			self.loglikes.append(self.loglike())
			print "Iteration: %d" % (i+1)
			print "Log-Likelihood: %s" % self.loglikes[i+1]
			if abs(self.loglikes[i] - self.loglikes[i+1]) < self.tol:
				break
			print "----------------------------------"

# Load text
text_data = np.load("text.npy", allow_pickle=False).astype(int)
with open('words.txt', 'r') as f:
	word_dict_lines = f.readlines()

# Process data
data = np.zeros((30799, 72406,))
for row in np.load("text.npy", allow_pickle=False).astype(int):
	doc_idx = row[0] - 1
	word_idx = row[1] - 1
	data[word_idx, doc_idx] = row[2]

LDAClassifier10 = LDA(num_topics=10, text_data=data, processed=True, words='words.txt', num_words=30799)
LDAClassifier5 = LDA(num_topics=5, text_data=data, processed=True, words='words.txt', num_words=30799)
LDAClassifier15 = LDA(num_topics=15, text_data=data, processed=True, words='words.txt', num_words=30799)

LDAClassifier10.fit()
LDAClassifier5.fit()
LDAClassifier15.fit()

LDAClassifier10.print_topics(n_words=50)
LDAClassifier5.print_topics(n_words=50)
LDAClassifier15.print_topics(n_words=50)

plt.plot(LDAClassifier10.loglikes[1:], '-')
plt.title('Objective-Function: Number of Topics = 10')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.show()

plt.plot(LDAClassifier5.loglikes[1:], '-')
plt.title('Objective-Function: Number of Topics = 5')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.show()

plt.plot(LDAClassifier15.loglikes[1:], '-')
plt.title('Objective-Function: Number of Topics = 15')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.show()
