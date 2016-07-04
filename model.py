from __future__ import print_function, division
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from common_utils import get_logger
import sys
import numpy
import pprint

MODEL_SCOPE 	= "char_rnn"
DEVICE_SCOPE 	= "/gpu:0"
logger = get_logger()

class Model(object):

	def __init__(self, config):
		"""Init model from provided configuration
		
		Args:
		    config (dict): Model's configuration
		    	Should have:
				rnn_size: 	size of RNN hidden state
				num_layers: number of RNN layers
				rnn_type:	lstm, rnn, or gru
				batch_size:	batch size
				seq_length: sequence length
				grad_clip: 	Clip gradient value by this value
				vocab_size: size of vocabulary
				infer:		True/False, if True, use the predicted output
							to feed back to RNN insted of gold target output.
				is_train:	True if is training
		"""

		logger.info("Create model with options: \n{}".format(pprint.pformat(config)))
		self.rnn_size 	= config["rnn_size"]
		self.num_layers = config["num_layers"]
		self.rnn_type 	= config["rnn_type"]
		self.batch_size = config["batch_size"]
		self.seq_length = config["seq_length"]
		self.grad_clip  = config["grad_clip"]
		self.vocab_size = config["vocab_size"]
		self.infer 		= config["infer"]
		self.is_train   = config["is_train"]
		self.reuse 		= config["reuse"]

		if self.rnn_type == "rnn":
			cell_fn = rnn_cell.BasicRNNCell
		elif self.rnn_type == "gru":
			cell_fn = rnn_cell.GRUCell
		elif self.rnn_type == "lstm":
			cell_fn = rnn_cell.LSTMCell
		else:
			msg = "Rnn type should be either rnn, gru or lstm"
			logger.error(msg)
			sys.exit(msg)

		# Define the cell
		cell = cell_fn(self.rnn_size)
		# Create multiple layers RNN
		self.cell = cell = rnn_cell.MultiRNNCell([cell] * self.num_layers)

		self.input_data 	= tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
		self.targets 		= tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
		self.initial_state	= cell.zero_state(self.batch_size, tf.float32)

		with tf.variable_scope(MODEL_SCOPE, reuse=self.reuse):
			softmax_w 	= tf.get_variable("softmax_w", [self.rnn_size, self.vocab_size], )
			softmax_b 	= tf.get_variable("softmax_b", [self.vocab_size])
			
			# Model params stored in DEVICE_SCOPE (here using GPU) 
			with tf.device(DEVICE_SCOPE):
				embeddings 	= tf.get_variable("embeddings", [self.vocab_size, self.rnn_size])

				# Split it into list of step input, i.e. along dimension 1
				inputs 		= tf.split(1, self.seq_length, tf.nn.embedding_lookup(embeddings, self.input_data))
				'''
				tf.split works like numply.split, inputs is now a list of step
				inputs (to rnn). Each step input has shape (batch_size, 1, rnn_size).
				We don't need that dimension 1, remove it by squeezing.
				'''
				inputs 		= [tf.squeeze(_input, [1]) for _input in inputs]

			'''
			Instead of writing the neuralnet manually, use seq2seq.rnn_decoder.
			In test time, the predicted output is fed back to RNN instead of 
			gold target output like in training time.
			'''
			def loop(prev, _):
				prev = tf.matmul(prev, softmax_w) + softmax_b
				# Wow, this stop_gradient is cool
				prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
				return tf.nn.embedding_lookup(embeddings, prev_symbol)

			outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if self.infer else None, scope=MODEL_SCOPE)
			# Concat each sequence of the batch 
			output = tf.reshape(tf.concat(1, outputs), [-1, self.rnn_size]) # now (batch_size x seq_length) x rnn_size 
			self.logits = tf.matmul(output, softmax_w) + softmax_b
			self.probs 	= tf.nn.softmax(self.logits)
			loss 		= seq2seq.sequence_loss_by_example( \
							[self.logits], \
							[tf.reshape(self.targets, [-1])], \
							[tf.ones([self.batch_size * self.seq_length])])
			self.cost  			= tf.reduce_sum(loss) / (self.batch_size * self.seq_length)
			self.final_state	= last_state

			if not self.is_train:
				return

			self.lr 	= tf.Variable(0.0, trainable=False)
			tvars 		= tf.trainable_variables()
			grads, _ 	= tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)

			optimizer 		= tf.train.AdamOptimizer(self.lr)
			self.train_op 	= optimizer.apply_gradients(zip(grads, tvars))

	def sample(self, sess, chars, vocab, num=500, prime=" ", sampling_type=1):
		state = self.cell.zero_state(1, tf.float32).eval()
		for char in prime[:-1]:
			x = numpy.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
			[state] = sess.run([self.final_state], feed)

		def weighted_pick(weights):
			t = numpy.cumsum(weights)
			s = numpy.sum(weights)

			return int(numpy.searchsorted(t, numpy.random.rand(1) * s))

		ret 	= prime
		char 	= prime[-1]
		for n in range(num):
			x = numpy.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
			[probs, state] = sess.run([self.probs, self.final_state], feed)
			p = probs[0]

			if sampling_type == 0:
				sample = numpy.argmax(p)
			elif sampling_type == 2:
				if char == " ":
					sample = weighted_pick(p)
				else:
					sample = numpy.argmax(p)
			else:
				sample = weighted_pick(p)

			pred = chars[sample]
			ret += pred
			char = pred 

		return ret
