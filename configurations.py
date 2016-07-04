def get_config_kimdung():
	config = {}
	config["rnn_size"] 		= 256
	config["num_layers"]	= 3
	config["rnn_type"]		= "gru"
	config["batch_size"]	= 80
	config["seq_length"]	= 80
	config["num_epochs"]	= 100
	config["grad_clip"]		= 5.
	config["lr"]			= 0.002
	config["decay_rate"]	= 0.97
	config["save_to"]		= "./saved_models/kimdung"
	config["data_dir"]		= "./data/kimdung"
	config["train_data"]	= "./data/kimdung/train.txt"
	config["dev_data"]		= "./data/kimdung/dev.txt"

	return config