import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import random

"""
###### Input: PMF of a target distribution，P(Y=y_i) ######
    bin_centers: y_i, np.1Darray
    histogram: P(Y=y_i), np.1Darray
    numSamples: the number of samples
    curS: current state, w.r.t. y_i
###### Output: list of states (y_i's) ######
"""
def plot_mcmc(bin_centers, histogram, numSamples, curS):
    searchIdx = range(len(bin_centers))
    states = [curS]
    for i in (range(numSamples)):
        nextS = np.random.choice(searchIdx,1)
        u = random.random()
        if u < np.min( [histogram[nextS] / histogram[curS],1] ):
            states.append(bin_centers[nextS])
            curS = nextS
        else:
            states.append(bin_centers[curS])
    return states

if __name__ == "__main__":
    ########### Gegerate data (for test)  ###########
    np.random.seed(0)
    # Sample from a normal distribution using numpy's random number generator
    # samples = np.random.normal(size=10**5)
    samples = np.random.beta(2,8,size=10**4)

    # Compute a histogram of the sample
    bins = np.linspace(0, 1, 50)
    histogram, bins = np.histogram(samples, bins=bins, density=True)

    bin_centers = 0.5*(bins[1:] + bins[:-1])

    # Compute the PDF on the bin centers from scipy distribution object
    pdf = stats.beta.pdf(bin_centers,2,8)

    # plt.figure(figsize=(6, 4))
    # plt.plot(bin_centers, histogram, label="Histogram of samples")
    # plt.plot(bin_centers, pdf, label="PDF")
    # plt.legend()
    # plt.show()

    ########### Get samples  ###########
    np.random.seed(1)
    numSamples = 10**5
    searchIdx = range(len(bin_centers))
    curS = np.random.choice(searchIdx,1)
    states = plot_mcmc(bin_centers, histogram, numSamples, curS)

    ########### Plot ###########
    plt.hist(np.array(states),bin_centers,density=True,  label="MCMC")
    plt.plot(bin_centers, histogram, label="Histogram of samples")
    plt.plot(bin_centers, pdf, label="PDF")
    plt.legend()
    plt.show()