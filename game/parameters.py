# BOARD
BOARD_SIZE = 4

# ANET
LEARNING_RATE = 1e-3
ACTIVATION_FUNCTION = 'relu' # linear, sigmoid, tanh, RELU
OPTIMIZER = 'adam' # Adagrad, SGD (Stochastic Gradient Descent), RMSProp, Adam
NUM_NEURONS = [8, 16, 8] # from i=0, to i=NUM_HIDDEN_LAYERS. I.e., NUM_HIDDEN_LAYERS+1 elements

# MCTS
NUM_EPISODES = 50
NUM_SEARCH = 10*BOARD_SIZE**2
EXPLORATION_WEIGHT = 1.0
EPSILON = 1.0
EPSILON_DECAY = 0.95
NUM_GAMES = 15
SIMULATE_GAMES = 'single' # single, multi

# TOPP
NUM_CACHED_ANETS = 5 # (M)
NUM_GAMES = 10 # (G)

# Policy network parameters
L2_REGULARIZATION = 1e-6 # Set to 0 for no regularization
BATCH_SIZE = 2048
NUM_EPOCHS = 20
LOG_INTERVAL = 5
SAVE_INTERVAL = 5


# # For training large networks from scratch
# # BOARD
# BOARD_SIZE = 7

# # ANET
# LEARNING_RATE = 1e-3
# ACTIVATION_FUNCTION = 'relu' # linear, sigmoid, tanh, RELU
# OPTIMIZER = 'adam' # Adagrad, SGD (Stochastic Gradient Descent), RMSProp, Adam
# NUM_NEURONS = [8, 16, 16, 8] # from i=0, to i=NUM_HIDDEN_LAYERS. I.e., NUM_HIDDEN_LAYERS+1 elements

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