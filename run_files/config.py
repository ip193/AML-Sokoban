import torch
dtype = torch.FloatTensor
cuda = False

# dtype = torch.cuda.FloatTensor
# cuda = True

FILE_TRIES = 3  # retry saving/loading in case of access conflicts
SLEEP_TIME = 3  # seconds until retry