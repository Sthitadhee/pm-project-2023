import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import pytensor.tensor as pt
from scipy.stats import norm, beta
from scipy.special import logit, expit
import time
from scipy.special import gamma
import matplotlib.pyplot as plt
import os

np.set_printoptions(suppress=True)

def phi(x):
    return norm.cdf(x)

def betaFunction(y, mu, precision, epsilon=1e-2):
    phi = precision
    power1 = phi * mu - 1
    power2 = phi * (1-mu) - 1
    divider = gamma(mu * phi) * gamma((1-   mu) * phi) + epsilon
    return (gamma(phi) * pow(y, power1) * pow(1-y, power2)) / (divider+epsilon)

def create_pi0(theta3, theta4, v):
    # expit == inv_logit
    return expit(theta3 + theta4 * v)

def create_pi1(theta5, theta6, v):
    return expit(theta5 + theta6 * v)
    
def create_mu_y(theta1, theta2, v):
    return phi(np.log(v/theta1) * (1/theta2))

def noise(d):
    return np.log(d/(1-d))

def get_mu_sd(theta1, theta2, lower_mu, prob_lower_mu, upper_mu, prob_upper_mu):
    # formulae for calculate the wind speed for the lower mu
    # we need this 
    # exp((qnorm(x) - (-1 / priors_beta_mu0) * log(priors_beta_mu1 / norm_factor)) * priors_beta_mu0)
    wind_speed_lower_mu = np.exp(norm.ppf(lower_mu) * theta2 + np.log(theta1))
    # log(p / (1 - p)) convert to the real number line
    logit_lower_mu = logit(prob_lower_mu)
    # repeat for upper mu
    wind_speed_upper_mu = np.exp(norm.ppf(upper_mu) * theta2 + np.log(theta1))
    logit_upper_mu = logit(prob_upper_mu)

    coeffs = np.array([[1, wind_speed_lower_mu], [1, wind_speed_upper_mu]])
    mus = np.array([logit_lower_mu, logit_upper_mu])
    thetas = np.linalg.solve(coeffs, mus)

    return thetas[0], thetas[1]


if __name__ == "__main__":

    # create thetas based on paper
    params = {
        'low': {
            'theta1' : [220, 3.33],
            'theta2' : [0.15, 0.02],
            'theta3' : [88, 8.33],
            'theta4' : [-0.57, 0.06],
            'theta5' : [-7.8, 1.2],
            'theta6' : [0.025, 0.006],
            'precision' : [3, 0.3],
        },
        'medium': {
            'theta1' : [270, 5],
            'theta2' : [0.36, 0.023],
            'theta3' : [13, 2.5],
            'theta4' : [-0.07, 0.013],
            'theta5' : [-68, 5.9],
            'theta6' : [0.22, 0.02],
            'precision' : [6, 0.6]
        },
        'high': {
            'theta1' : [320, 10],
            'theta2' : [0.265, 0.02],
            'theta3' : [65, 5.9],
            'theta4' : [-0.283, 0.03],
            'theta5' : [-90, 12],
            'theta6' : [0.205, 0.024],
            'precision' : [9.7, 2.1],
        },
    }

    # create thetas based on code
    building_quality = 'low'
    # the highest wind speed velocity
    norm_factor = 1

    theta1_ = params[building_quality]['theta1'][0]
    theta2_ = params[building_quality]['theta2'][0]
    # theta3_ = params[building_quality]['theta3'][0]
    # theta4_ = params[building_quality]['theta4'][0]
    # theta5_ = params[building_quality]['theta5'][0]
    # theta6_ = params[building_quality]['theta6'][0]

    precision_ = params[building_quality]['precision'][0]

    theta1_ = theta1_/norm_factor

    # calculate the appropiate mu_theta3, mu_theta4 for pi_0
    theta3_, theta4_ = get_mu_sd(theta1_, theta2_, 0.01, 0.99, 0.05, 0.01)
    
    # calculate the appropiate mu_theta5, mu_theta6 for pi_1
    theta5_, theta6_ = pi_1_params = get_mu_sd(theta1_, theta2_, 0.95, 0.01, 0.99, 0.99)

    v = np.arange(5, 500, 5)
    v = np.repeat(v, 5)/norm_factor

    pi0s = create_pi0(theta3_, theta4_, v)
    pi1s = create_pi1(theta5_, theta6_, v)
    # the expected value of the ys
    mus = create_mu_y(theta1_, theta2_, v)
    probs = np.random.beta(mus * precision_, (1-mus) * precision_)

    damage_estimates = []
    for i in range(pi0s.shape[0]):
        prob_array = [pi0s[i], (1 - pi0s[i]) * (1 - pi1s[i]), (1 - pi0s[i]) * pi1s[i]]
        possible_values = [0, probs[i], 1]
        damage_estimates.append(np.random.choice(possible_values, p=prob_array))

    damage_estimates = np.array(damage_estimates)

    figure, ax = plt.subplots()
    ax.plot(v, damage_estimates, 'k-')
    ax.set_xlabel('v', fontsize = 14)
    ax.set_ylabel('mu', fontsize = 14)

    # damage_estimates.size
    absolute_path = os.path.dirname(os.path.dirname(__file__))
    path = f'{absolute_path}/assets/data/synthetic_data.csv'
    print(path)
    df = pd.DataFrame({'x': v, 'y': damage_estimates})
    df.to_csv(path, index=False)


