from __future__ import print_function, division

import os 
import time
import pprint
import argparse
import configurations 
import tensorflow as tf
from model import Model
from reader import DataFeeder
from common_utils import get_logger
import numpy

logger = get_logger()

parser = argparse.ArgumentParser()
parser.add_argument(
	"--config", default="get_config_kimdung",
	help="Configuration function defined in configurations.py")
parser.add_argument(
	"--mode", choices=["train", "generate"], default="train",
	help="Mode to run. Default to train a model. If 'generate', require a model.")
parser.add_argument(
	"--model-dir", 
	help="Folder to existing model, used to generate text.")
args = parser.parse_args()

def run_epoch(sess, m, data_feeder, eval_op, config, e):
	state = m.initial_state.eval()
	costs = 0.0
	iters = 0

	for b in range(data_feeder.num_batches):
		start = time.time()
		x, y = data_feeder.next_batch()
		feed = {m.input_data: x, m.targets: y, m.initial_state: state}
		cost, state, _ = sess.run([m.cost, m.final_state, eval_op], feed)
		end = time.time()

		costs += cost 
		iters += m.rnn_size

		if b % 100 == 0:
			logger.info("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
			                    .format(e * data_feeder.num_batches + b,
			                            config["num_epochs"] * data_feeder.num_batches,
			                            e, cost, end - start))

	return numpy.exp(costs / iters)

if __name__ == "__main__":

	if args.mode == "train":
		# Get configuration
		train_config 	= getattr(configurations, args.config)()
		dev_config 		= getattr(configurations, args.config)()
		logger.info("Model options: \n{}".format(pprint.pformat(train_config)))

		# Create save_to folder if not exists
		if not os.path.exists(train_config["save_to"]):
			os.makedirs(train_config["save_to"])
		
		logger.info("Load training data from {}, batch_size = {}, seq_length = {}" \
			.format(train_config["train_data"], train_config["batch_size"], train_config["seq_length"]))
		logger.info("Load dev data from {}, batch_size = {}, seq_length = {}" \
			.format(dev_config["dev_data"], dev_config["batch_size"], dev_config["seq_length"]))
		train_data_feeder 	= DataFeeder(train_config["train_data"], 
											train_config["batch_size"], 
											train_config["seq_length"])
		dev_data_feeder 	= DataFeeder(dev_config["dev_data"], 
											dev_config["batch_size"], 
											dev_config["seq_length"], vocab=train_data_feeder.vocab)

		train_config["is_train"] = True
		train_config["infer"]	 = False
		train_config["reuse"]	 = False
		dev_config["is_train"]	 = False
		dev_config["infer"]	 	 = True
		dev_config["reuse"]	     = True
		train_config["vocab_size"] 	= dev_config["vocab_size"] 	= len(train_data_feeder.vocab)

		with tf.Graph().as_default(), tf.Session() as sess:
			initializer = tf.random_uniform_initializer()
			train_model = Model(train_config) 
			dev_model 	= Model(dev_config)

			tf.initialize_all_variables().run()
			# Start saver after init all vars
			saver = tf.train.Saver(tf.all_variables()) 
			
			best_ppl = 10e5			
			for e in range(train_config["num_epochs"]):
				logger.info("-" * 30)
				start = time.time()
				sess.run(tf.assign(train_model.lr, train_config["lr"] * (train_config["decay_rate"] ** e)))
				train_data_feeder.reset_pointer()
				dev_data_feeder.reset_pointer()

				logger.info("Epoch: {}, learning rate = {:.3f}".format(e + 1, sess.run(train_model.lr)))
				train_ppl = run_epoch(sess, train_model, train_data_feeder, train_model.train_op, train_config, e)
				logger.info("Epoch: {}, train perplexity = {:.3f}".format(e + 1, train_ppl))
				end = time.time()
				logger.info("Time/epoch {}".format(end - start))
				logger.info("\n")

				start = end
				dev_ppl = run_epoch(sess, dev_model, dev_data_feeder, tf.no_op(), dev_config, e)
				logger.info("Epoch: {}, dev perplexity = {:.3f}".format(e + 1, dev_ppl))
				end = time.time()
				logger.info("Time/epoch {}".format(end - start))

				if dev_ppl < best_ppl:
					best_ppl = dev_ppl
					# Save best model
					checkpoint_path = os.path.join(train_config["save_to"], "model.ckpt")
					saver.save(sess, checkpoint_path, global_step=e)
					logger.info("Model saved to {}".format(checkpoint_path))
