from scipy.stats import norm
from scipy.integrate import quad
import numpy as np

# Define the conditional probability integral
def conditional_integrand(g1,a,b,mu=0,dev=1.5):
    if a==-np.inf:
        return norm.cdf((b - mu - 0.5*g1) / dev) * norm.pdf(g1)
    elif b==np.inf:
        return (1-norm.cdf((a - mu - 0.5*g1) / dev)) * norm.pdf(g1)
    else:
        return (norm.cdf((b - mu - 0.5*g1) / dev) - norm.cdf((a - mu - 0.5*g1) / dev))* norm.pdf(g1)

inter=[-np.inf,-0.8416,-0.2533,0.2533,0.8416,np.inf]
for k in range(5):
    aa=inter[k]
    bb=inter[k+1]
    sum=0
    print(f'denomitore [{aa},{bb}]')
    for h in range(5):
        a=inter[h]
        b=inter[h+1]

        numerator, _ = quad(lambda g1: conditional_integrand(g1,a,b), aa, bb)
    # Calculate the denominator: P(-0.25 <= G1 <= 0.25)
        if aa==-np.inf:
            denominator = norm.cdf(bb)
        elif bb==np.inf:
            denominator = 1-norm.cdf(aa)
        else:
            denominator = norm.cdf(bb)-norm.cdf(aa)

    # Compute the conditional probability
        conditional_probability = numerator / denominator
        sum=conditional_probability+sum
        print(f'numeratore[{a},{b}]')
        print(conditional_probability)
        
    print(sum)
    print('\n')
