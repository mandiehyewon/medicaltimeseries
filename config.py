# config
config = lambda: None

# Experiment setup
config.model_name = "lstm" 
config.l2_coeff = 0.002 
config.dropout = 0.25

config.tensorboard = False


config.task = 0

# architecture
config.input_dim = 35
config.num_layers = 2
config.hidden_dim = 100
config.output_dim = 1

config.num_samples = 10

# training config
config.l2_coeff = 0.002
config.NUM_EPOCHS = 2000
config.BATCH_SIZE = 128
config.LR = 1e-3
config.decay_rate = 0.1
config.decay_iter = 56000
