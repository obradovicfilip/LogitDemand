import numpy as np
import pandas as pd
import logit_demand.LogitDemand as lg

"""This script is used to test the logit_demand functions"""

# These functions are ONLY for making simulated data
def choice_probability(I, X ,beta, normalize=1):
    scores = np.exp(X @ beta)
    d = {'consumers': I, 'scores': scores}
    df = pd.DataFrame(data=d)
    if normalize == 1:
        df['maxs'] = df.groupby('consumers')['scores'].transform('max')
        df['scores'] = df['scores'] / df['maxs']
        scores = df.scores.values
    df['norms'] = df.groupby('consumers')['scores'].transform('sum')
    norms = df.norms.values
    probabilities = scores / norms
    return probabilities


def loglikelihood(Y,I,X,beta):
    probabilities =choice_probability(I, X ,beta, normalize=1)
    l = -sum(Y * probabilities)
    return l

# Settings
# pd.set_option('display.max_columns', None) #Display all columns in pandas

N = 1000
M = 6
C = [3,6]
TC = 100
seed = 750
np.random.seed(seed)
use_data = True
boot_samples = 100


if use_data == False:
# Make a mock dataset
    consumers = np.arange(0,1000)
    locs = np.random.uniform(-50,50,M-1)
    stds = np.random.lognormal(0,5,M-1)
    p_attrs = np.random.multivariate_normal(locs, np.diag(stds),TC)
    p = np.random.lognormal(5, 3, TC)
    c_attrs = np.random.binomial(1,0.5,(N,int(np.floor((M-1)/2))))
    offered = [np.random.choice(TC, size=np.random.randint(*C), replace=False, p=None) for i in range(N)]
    offered_l = list(map(len,offered))
    I = [np.tile(i,offered_l[i]) for i in range(N)] #Tile consumers according to the number of offered products
    I = np.concatenate(I, axis=None) #Concatenate into one array
    # I = np.stack((consumers, offered), axis = -1) #This is old
    # print("Offered_l is: "+str(offered_l[:10])) Debugging code
    # print("I is: "+ str(I[:10]))

    products = np.c_[p, p_attrs]
    Xs = [products[offered[i]] for i in range(N)] # List of X matrices for each consumer - initialize
    for i in range(N):
        Xs[i][:, -int(np.floor((M-1)/2)):] *= c_attrs[i] # Change the last floor (M-1)/2 columns to correspond to c_attrs
    Xs = np.concatenate(Xs, axis=0) #Make a matrix of consumer-product attributes for each consumer and offered product (conformal to I)

    #beta = np.random.uniform(-0.2,0.2,(1,M)).ravel() #Draw betas
    beta = [0.5]*M
    beta[0] = -abs(beta[0]) #Make the first beta negative
    probabilities = choice_probability(I,Xs,beta)
    counter = 0
    sampled_product = []
    for i in range(N):
        sampled_product.append(np.random.choice(offered[i],p=probabilities[counter:counter+offered_l[i]]))
        counter+=offered_l[i] #This samples a product out of the offered ones for each consumer and stacks it into a vector

    split_probabilities = np.split(probabilities, np.cumsum(offered_l))[:-1] #This sections probabilities to conform to shape of offered - aids in searching for prob
     # sampled_product = [np.random.choice(offered[i],p=probabilities) for i in range(N)] #Old version
    Y = [np.tile(0,offered_l[i]) for i in range(N)] #Tile choices according to the number of offered products. We will only let them potentially choose the highest score product
    for i in range(N):
        Y[i][np.asscalar(np.where(offered[i]==sampled_product[i])[0])] = np.random.binomial(1,p=split_probabilities[i][np.asscalar(np.where(offered[i]==sampled_product[i])[0])])#For each consumer draws 0 or 1 based on the probability of the sampled product
    Y = np.concatenate(Y, axis=None)
    data = np.c_[I, Y, Xs]
    data = pd.DataFrame(data)
    data = data.rename(columns={0:'consumer_id',1:'Y',2:'price',3:'age',4:'female',5:"num_dependents",6:'public',7:'older'})
    log = lg.LogitDemand(data, index_column='consumer_id')
    beta = log.fit([-1,1,1,1,1,1],boot_samples=boot_samples)
    print("Data used was mock")
else:
    data = pd.read_hdf(r'..\data\clean_data.hdf', key='data')
    log = lg.LogitDemand(data[:10000])
    beta = log.fit([-1,1,1,1,1,1],boot_samples=boot_samples)
    print(beta)
    print("Betas are "+str(beta.x))
    print("Probabilities are:" + str(log.choice_probability(beta.x)))
    print("Likelihood is: " + str(log.loglikelihood(beta.x)))
    log.plot_fit(beta.x)
    print("Data used was real")


