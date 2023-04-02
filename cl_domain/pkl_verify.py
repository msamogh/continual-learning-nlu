import pickle
from cl_domain.domain import *

o = pickle.load(open("../cl_runs/random-legacy-coda/run_0.pkl", "rb"))
print([len(split.train_samples) for split in o.domain_wise_splits.values()])
print([len(split.val_samples) for split in o.domain_wise_splits.values()])
print([len(split.test_samples) for split in o.domain_wise_splits.values()])