#Example to buiild on
from joblib import Parallel, delayed
import time
import numpy
import pandas as pd
import logit_demand.LogitDemand as lg

from joblib import Parallel, delayed

def square(x):
    return x*x
tic = time.perf_counter()
policy=Parallel(n_jobs=24)(delayed(square)(i) for i in range(0,10))
toc = time.perf_counter()
print(toc-tic)

# tic = time.perf_counter()
# for i in range(10):
#     data = pd.read_hdf(r'..\data\clean_data.hdf', key='data')
#     log = lg.LogitDemand(data[:10000])
#     # beta = log.fit([0,0,0,0,0,0])
#     beta = log.fit([-1, 1, 1, 1, 1, 1])
#     print("Betas are " + str(beta.x))
#     print("Probabilities are:" + str(log.choice_probability(beta.x)))
#     print("Likelihood is: " + str(log.loglikelihood(beta.x)))
#     log.plot_fit(beta.x)
#     print("Data used was real")
#
# toc = time.perf_counter()
# print(toc-tic)


import time
import numpy
import pandas as pd
import logit_demand.LogitDemand as lg
data = pd.read_hdf(r'..\data\clean_data.hdf', key='data')
log = lg.LogitDemand(data[:10000])
tic = time.perf_counter()
policy=Parallel(n_jobs=24)(delayed(log.fit)([-1, 1, 1, 1, 1, 1]) for i in range(0,10))
toc = time.perf_counter()
print(toc-tic)

print(policy[1].x)