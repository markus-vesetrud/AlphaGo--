# BOARD
BOARD_SIZE = 4

# ANET
LEARNING_RATE         = 1e-3
ACTIVATION_FUNCTION   = 'relu' # linear, sigmoid, tanh, RELU
OPTIMIZER             = 'adam' # Adagrad, SGD (Stochastic Gradient Descent), RMSProp, Adam
NUM_NEURONS           = [8, 12, 8] # from i=0, to i=NUM_HIDDEN_LAYERS. I.e., NUM_HIDDEN_LAYERS+1 elements
DROPOUT_PROB          = 0.25

# MCTS
NUM_EPISODES = 40
NUM_SEARCH = 100
EXPLORATION_WEIGHT = 1.0
EPSILON = 1.0
EPSILON_DECAY = 0.96
NUM_GAMES = 1
SIMULATE_GAMES = 'single' # single, multi

# TOPP
NUM_CACHED_ANETS = 5 # (M)
TOPP_NUM_GAMES = 100 # (G)

# Policy network parameters
L2_REGULARIZATION = 2e-5 # Set to 0 for no regularization
BATCH_SIZE = 100
NUM_EPOCHS = 15

# RL model and replay buffer
REPLAY_BUFFER_MAX_LENGTH = 400
INITIAL_REPLAY_BUFFER    = None # '4by4_1000_iter_20_games.npy' # path or None
MODEL_START              = None # path or None

VISUALIZE_GAMES_RL = False
VISUALIZE_GAMES_TOPP = False

# # For training large networks from scratch
# # BOARD
# BOARD_SIZE = 7

# # ANET
# LEARNING_RATE = 1e-3
# ACTIVATION_FUNCTION = 'relu' # linear, sigmoid, tanh, RELU
# OPTIMIZER = 'adam' # Adagrad, SGD (Stochastic Gradient Descent), RMSProp, Adam
# NUM_NEURONS = [8, 16, 16, 8] # from i=0, to i=NUM_HIDDEN_LAYERS. I.e., NUM_HIDDEN_LAYERS+1 elements
# INITIAL_REPLAY_BUFFER = '7by7_1470_iter_150_games.npy' # path or None
# MODEL_START           = 'test_model.pt' # path or None

# # MCTS
# NUM_EPISODES = 200
# NUM_SEARCH = 15*BOARD_SIZE**2
# EXPLORATION_WEIGHT = 1.0
# EPSILON = 1.0
# EPSILON_DECAY = 0.99
# NUM_GAMES = 15
# SIMULATE_GAMES = 'multi' # single, multi

# # TOPP
# NUM_CACHED_ANETS = 5 # (M)
# NUM_GAMES = 10 # (G)

# # Policy network parameters
# L2_REGULARIZATION = 1e-6 # Set to 0 for no regularization
# BATCH_SIZE = 2048
# NUM_EPOCHS = 20
# LOG_INTERVAL = 5
# SAVE_INTERVAL = 5