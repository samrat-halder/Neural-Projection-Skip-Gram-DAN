from config import *
import pickle
import os
import random


def gen_seed(idx, n=100):
  random.seed(12345)

  seeds = []
  for i in range(n):
    s = random.random()
    seeds.append(s)

  return seeds[idx]

def dump_pickle(obj, data_dir, fname):
    with open(os.path.join(data_dir, fname), 'wb') as f:
        pickle.dump(obj, f)
    f.close()

def load_pickle(data_dir, fname):
    with open(os.path.join(data_dir, fname), 'rb') as f:
        obj = pickle.load(f)
    return obj

