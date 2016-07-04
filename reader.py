from __future__ import print_function, division
import os
from codecs import open
import numpy
import cPickle as pickle
import sys
from common_utils import get_logger

logger = get_logger()

class DataFeeder(object):
	"""Create vocabulary and batch iterator.
	"""
	def __init__(self, file_path, batch_size, seq_length, vocab=None):
		"""DataFeeder class. 
		Take a data file, create batches (input, output) of (batch_size x seq_length)
		and provide iterator over the batches.
		
		Args:
		    file_path (string): Path to data file (text)
		    batch_size (Number): Integer, size of batch
		    seq_length (Number): Integer, length of sequence
		    vocab (None, optional): Python dictionary object, model's dictionary
		"""
		self.file_path 		= file_path
		self.batch_size 	= batch_size
		self.seq_length 	= seq_length
		self.tensor_path 	= file_path + ".npy"
		self.vocab_path 	= file_path + ".vocab.pkl"

		if not os.path.exists(self.file_path):
			msg = "Data file not exists"
			logger.error(msg)
			sys.exit(msg)

		self.vocab 	= vocab
		self.tensor = None
		self.x_batches = None
		self.y_batches = None
		self.pointer = 0

		self.load_data()
		self.create_batches()

	def load_data(self):
		"""Load saved data and vocab, if not exist then create them from data 
		text file.
		"""
		if not os.path.exists(self.tensor_path) or not os.path.exists(self.vocab_path):
			logger.info("Vocab & data file not exist, process now!")
			self.process()
		else:
			logger.info("Load vocab & data from files.")
			with open(self.vocab_path, "rb") as f:
				self.vocab = pickle.load(f)
			self.tensor = numpy.load(self.tensor_path)

	def process(self):
		"""Load data to numpy array, create vocabulary and save both to files.
		"""
		with open(self.file_path, "r", "utf-8") as f:
			data = list(f.read())

			# Save data tensor + vocab to files
			open(self.tensor_path, "w").close()
			open(self.vocab_path, "w").close()

			char_set = set(data)
			self.vocab = self.vocab if self.vocab else {ch: idx for idx, ch in enumerate(char_set)}
			with open(self.vocab_path, "wb") as f:
				pickle.dump(self.vocab, f)

			default_ch_idx = self.vocab.get(" ", 0)
			self.tensor = numpy.array([self.vocab.get(ch, default_ch_idx) for ch in data])
			numpy.save(self.tensor_path, self.tensor)

	def create_batches(self):
		"""Split data into batches
		"""
		self.num_batches = self.tensor.shape[0] // (self.batch_size * self.seq_length)

		if self.num_batches == 0:
			msg = "Not enough data (num_batches = 0)!"
			logger.error(msg)
			sys.exit(msg)

		self.tensor = self.tensor[:(self.num_batches * self.batch_size * self.seq_length)]
		x_data = self.tensor
		y_data = numpy.copy(x_data)
		y_data[:-1] = x_data[1:]
		y_data[-1] 	= x_data[0]

		self.x_batches = numpy.split(x_data.reshape(self.batch_size, -1), self.num_batches, -1)
		self.y_batches = numpy.split(y_data.reshape(self.batch_size, -1), self.num_batches, -1)

	def next_batch(self):
		"""Get next batch
		"""
		self.pointer += 1
		return self.x_batches[self.pointer - 1], self.y_batches[self.pointer - 1]

	def reset_pointer(self):
		"""Reset pointer to next batch
		"""
		self.pointer = 0


if __name__ == "__main__":
	data_feeder = DataFeeder("./data/old_kimdung/train.txt", 80, 80)
	print(data_feeder.num_batches)
	ivocab = {idx: ch for ch, idx in data_feeder.vocab.items()}
	for i in xrange(data_feeder.num_batches):
		x, y = data_feeder.next_batch()
		print(x, y)
		# for seq in x:
		# 	print(u"".join(map(ivocab.get, seq)))
