# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os 
import sys
import cPickle as pickle
import tensorflow as tf
from model import Model
from six import text_type
import codecs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", required=True,
        help="Where model checkpoint, config.pkl, model.ckpt are saved.")
    parser.add_argument(
        "--vocab_path", required=True,
        help="Path to vocab.pkl.")
    parser.add_argument("--length", type=int, default=500,
                       help="number of characters to sample")
    parser.add_argument("--prime", type=text_type, default=u"Thiên hạ ",
                       help="prime text")
    parser.add_argument("--sampling_type", type=int, default=1,
                       help="0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces")
    args = parser.parse_args()
    sample(args)

def sample(args):
    config_path = os.path.join(args.save_dir, "config.pkl")
    if not os.path.exists(config_path):
        sys.exit("config.pkl not exists!")
    if not os.path.exists(args.vocab_path):
        sys.exit("vocab.pkl not exists!")

    with open(config_path, "rb") as f:
        config = pickle.load(f)
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    config["infer"] = True
    model = Model(config)
    ivocab = {idx: ch for ch, idx in vocab.items()}
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            text = model.sample(sess, ivocab, vocab, num=args.length, prime=args.prime, sampling_type=args.sampling_type)
            save_to = "./sample.txt"
            open(save_to, "w").close()
            with codecs.open(save_to, "w", "utf-8") as f:
                f.write(text)

if __name__ == "__main__":
    main()
