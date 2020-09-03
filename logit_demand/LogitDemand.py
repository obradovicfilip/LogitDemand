"""Logit demand module. Returns a logit demand object, fits it and bootstraps standard errors"""
"""Bugs:__INIT__.PY DOES NOT READ CONFIG. I think it should do it automatically according to the documentation,
but it will not function. Works without that."""

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import time
from scipy.optimize import minimize
from joblib import Parallel, delayed
from random import choices

# Initialize global variable that will be written over to facilitate parallelization
cols = []
data = []

class LogitDemand(object):
    """
    Logit demand class, returns a logit demand object when provided with a dataframe.

    Methods:
    choice_probability(self, beta, normalize = 1) - returns a choice probability for each choice.

    loglikelihood(self,beta) - returns the log likelihood for a particular vector of betas.

    plot_fit(self,beta) - plots a distribution of choice probabilities for a particular vector of betas.

    fit(self, initial_guess, neg_constraint=True,quiet=False) - Finds the MLE for betas. Returns bootstrapped standard
    errors.
    """

    def __init__(self,df, columns=['price', 'age', 'female', 'num_dependents', 'public', 'older'], index_column='index'):
        """
        Constructor method
        Keyword arguments:
        columns -- set the name of columns that constitute X
                   (default = ['price', 'age', 'female', 'num_dependents', 'public', 'older'])
        index_column -- set the name of the index column in the dataset. (default = 'index')
        """
        # Reading from config file determines whether to omit males and those older than 65 - NOT WORKING
        try:
            if female_only == 1:
                df = df[df['female'] == 1]
        except:
            pass
        try:
            if under_65 ==1:
                df = df[df['age']<65]
        except:
            pass
        df = df.rename(columns={index_column:'index'}) # Rename the index column so that code runs
        # Load data
        self.I = df['index'].values # Define the consumers. Each consumer has multiple decisions, treat each separately
        self.X = df[columns].values # Matrix of patient attributes
        self.Y = df['Y'].values # Vector of choices
        self.columns = columns # Save vector of chosen attributes
        global cols # Write columns to a global variable to avoid parallelization issues. Parallel fails
        cols =  self.columns # to activate when reading class variables!
        global data # Write the dataframe to a global variable again due to parallelization
        data = df

        # Test if a consumer picked two choices in a single decision (denoted by index)
        df['test'] = df.groupby('index')['Y'].transform('sum')
        if len(df[df['test']>1]!=0):
            raise ValueError('ERROR: Consumer made two choices instead of one!')
            print(len(df[df['test']>1]))
            print(df[df['test']>1])
        else:
            pass
        df.drop(columns=['test']) # Drop the test column

    def choice_probability(self, beta, normalize = 1):
        """
        Calculates the choice probability for every choice and consumer.
        Keyword arguments:

         normalize -- normalizes the scores of each choice for a particular consumer as a percent of the highest
                        scores. This helps with float number limitations that can occur due to the use of exponentials.
                        (default = 1)"""
        # Load necessary matrices
        I = self.I
        X = self.X
        scores = np.exp(X@beta) # Find scores
        d = {'consumers': I, 'scores': scores} # Making a dataframe
        df = pd.DataFrame(data=d)
        # If normalization is chosen, each choice is calculated as a percent of the highest score for each consumer
        if normalize == 1:
            df['maxs'] = df.groupby('consumers')['scores'].transform('max') # Find maximum score
            df['scores'] = df['scores']/df['maxs'] #Normalize
            scores = df.scores.values # Return normalized scores as a series
        df['norms'] = df.groupby('consumers')['scores'].transform('sum') # Find the sum of scores per consumer
        norms = df.norms.values # Extract series
        probabilities = scores / norms # Find choice probabilities
        return probabilities

    def loglikelihood(self,beta):
        """Returns the log likelihood for a particular vector of betas."""
        Y = self.Y
        X = self.X
        I = self.I
        probabilities = self.choice_probability(beta) # Call choice probabilities
        l = -sum(Y*probabilities) # Find the log likelihood
        return l


    def plot_fit(self,beta):
        """
        Plots a distribution of choice probabilities for a particular vector of betas.
        Saves it in the output folder
        """
        Y = self.Y # Load choices
        probabilities = self.choice_probability(beta) # Call choice probabilities
        probabilities = probabilities[Y==1] # Select only the chosen options
        sns_plot = sns.distplot(probabilities) # Draw the distribution
        fig = sns_plot.get_figure()
        fig.savefig("..\Output\choice_distribution.png") # Save plot


    def fit(self, initial_guess, neg_constraint=True, quiet=False, std_err='Bootstrap', boot_samples=1000, boot_sample_size=None):
        """
        Finds the MLE for betas. Returns estimates and bootstrapped standard errors.
        Keyword arguments:

        initial_guess -- Initial guess for the optimizing function. Must be the same dimension as beta.
        neg_constraint -- Imposes the non-negativity constraint on the first beta. This is supposed to be the price beta.
        quiet -- If True, will not print an output. (default = False)
        std_err -- If Bootstrap will return bootstrapped standard errors. (default = Bootstrap)
        boot_samples -- Set bootstrap sample number (default = 1000)
        boot_sample_size -- Set bootstrap sample sizes. If None, they will be 90% of the total sample size. (default = None)
        """
        # Check the dimensions of the initial guess. Raise error if incorrect.
        if len(initial_guess) != np.shape(self.X)[1]:
            raise AttributeError("Wrong dimensions of the initial guess. Dimension must be "+str(np.shape(self.X)[1])+".")

        # Check if non-negativity constraint is imposed on the price parameter and impose it in the optimizer if needed
        if neg_constraint == True:
            M = len(initial_guess) #Set the bounds on the first parameter
            bond = [(None, None)] * M # Make blank bounds
            bond[0] = (None, 0) # Set the non-negativity parameter bound
            estimates = minimize( fun = self.loglikelihood, x0 = initial_guess, bounds = bond) # Estimate
        else:
            bond = None
            estimates = minimize( fun = self.loglikelihood, x0 = initial_guess, bounds = None)
        if std_err == 'Bootstrap':
            betas = self._bootstrap_stderr(initial_guess = estimates.x, bounds = bond, samples=boot_samples,
                                  sample_size=boot_sample_size) # Find bootstrap estimates
            std = np.std(betas, axis=0) # Bootstrap standard errors
            stats = estimates.x/std
        else:
            std = "Standard errors not chosen"

        #Print results
        try:
            result_string = ["Beta_{k}   {est:.3f}   {st:.3f}   {z:.3f}".format(k=i,est=estimates.x[i],st=std[i],z=stats[i])
                            for i in range(len(estimates.x))]
        except:
            result_string = ["Beta_{k}   {est:.3f}   {st}   {st}".format(k=i, est=estimates.x[i], st="N/A")
                             for i in range(len(estimates.x))]
        result_string.insert(0,"")
        result_string[0] = 'Parameter'+"   "+"Estimate"+"   "+"S.E."+"   "+"Z"
        print("""---------------------------------------""")
        print('\n'.join(result_string))
        print("""---------------------------------------""")

        return estimates.x, std

    def _bootstrap_stderr(self, samples=1000, sample_size=None, initial_guess = None, bounds = None):
        """
        Method for bootstrapping the standard errors for betas. Refits the data for bootstrapped samples and returns
        the standard deviation of each beta. Uses all available CPU cores.

        samples -- Number of bootstrap samples (default = 1000)
        sample_size -- Size of each bootstrap sample. If None, each will have 90% of the total number of observations.
                       (default = None)
        initial_guess -- Initial guess for the optimizer function for fitting betas. If left as None it will take the
        estimated betas of the fit method as the initial_guess. (default = None)
        bounds -- Imposes bounds on beta estimation. If left as None, will take the same option as the fit method.
                  (default = None)
        """
        #Creating bootstrapped data
        def aux_func(df, initial_guess = initial_guess):
            """
            Auxiliary function for parallelization. Calculates the loglikelihood of a single bootstrapped
            sample
            """
            # Load data, same as in the constructor
            global cols # Load the global columns to avoid problems with parallelization
            I = df['index'].values
            X = df[cols].values
            Y = df['Y'].values
            def loglikelihood(beta = initial_guess, normalize=1):
                """
                Auxiliary loglikelihood function to avoid pulling from the method and dealing with pandas. Same as above.
                """
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
                l = -sum(Y * probabilities)
                return l
            res = minimize(fun=loglikelihood, x0=initial_guess, bounds=bounds)
            return res.x #Returns estimates only
        # Read the set bootstrap sample size
        if sample_size is None: # If left default
            effective_size = len(set(self.I)) #Take the number of consumers
            sample_size = int(0.9*effective_size) #Take 90% of the effective number of consumers to make bootstrap samples
        consumers = list(set(self.I)) # Read unique consumers (indices)
        boot = [[]]*samples
        for i in range(samples):
            boot[i] = choices(consumers, k=sample_size) #Make a sample of chosen consumers
        sample = [[]]*samples
        for i in range(samples):
            sample[i] = data[data['index'].isin(boot[i])] #Read all choices for chosen consumer and stack in a df
        betas = Parallel(n_jobs=-1)(delayed(aux_func)(sample[i]) for i in range(samples)) #Estimate the betas
        return betas

# Unit testing
if __name__=="__main__":
    data = pd.read_hdf(r'..\..\data\clean_data.hdf', key='data') #Load the data
    data.rename(columns={'index':'something_else'}) #Check if the code handles different index names
    log = LogitDemand(data[:20000]) #Initialize the object
    tic = time.perf_counter() #Measure execution time
    print(log.fit([-1, 1, 1, 1, 1, 1],boot_samples=24,std_err='Bootstrap')) #Fit
    toc = time.perf_counter()
    print("Running time was: "+str(toc-tic))