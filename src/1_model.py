import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import pytensor.tensor as pt
from scipy.stats import norm
from scipy.special import logit
import os
import graphviz
import scipy
import time

def zero_one_beta_logp(y, pi0, pi1, mu, precision):
    alpha = mu * precision
    beta = (1 - mu) * precision
    return pt.switch(pt.eq(y, 0),
                      pt.log(pi0),
                      pt.switch(pt.eq(y, 1),
                                pt.log((1 - pi0) * pi1),
                                pt.log((1 - pi0) * (1 - pi1)) + pm.logp(pm.Beta.dist(alpha=alpha, beta=beta), y)))


def get_mu_sd(lower_mu, prob_lower_mu, upper_mu, prob_upper_mu, building_type):
    theta1_mu, theta2_mu = theta1_theta2_mu_dict[building_type]

    wind_speed_lower_mu = np.exp(norm.ppf(lower_mu) * theta2_mu + np.log(theta1_mu/norm_factor))
    logit_lower_mu = logit(prob_lower_mu)

    wind_speed_upper_mu = np.exp(norm.ppf(upper_mu) * theta2_mu + np.log(theta1_mu/norm_factor))
    logit_upper_mu = logit(prob_upper_mu)

    coeffs = np.array([[1, wind_speed_lower_mu], [1, wind_speed_upper_mu]])
    mus = np.array([logit_lower_mu, logit_upper_mu])
    thetas = np.linalg.solve(coeffs, mus)
    return thetas[0], thetas[1]


def do_MCMC(thetas, wind_speeds, y):

    with pm.Model() as zoib:
        # Priors for unknown model parameters

        # precision = pm.Normal("precision", 100, 33) # based on the original code
        theta1 = pm.Normal("theta1", mu=thetas[0][0], tau=1/(thetas[0][1]**2))
        theta2 = pm.Normal("theta2", mu=thetas[1][0], tau=1/(thetas[1][1]**2))
        theta3 = pm.Normal("theta3", mu=thetas[2][0], tau=1/(thetas[2][1]**2))
        theta4 = pm.Normal("theta4", mu=thetas[3][0], tau=1/(thetas[3][1]**2))
        theta5 = pm.Normal("theta5", mu=thetas[4][0], tau=1/(thetas[4][1]**2))
        theta6 = pm.Normal("theta6", mu=thetas[5][0], tau=1/(thetas[5][1]**2))
        precision = pm.Uniform("precision", 0, 100)

        # Expected value of outcome
        PI_0 = pm.Deterministic("pi 0", pm.math.invlogit(theta3 + theta4 * wind_speeds))
        PI_1 = pm.Deterministic("pi 1", pm.math.invlogit(theta5 + theta6 * wind_speeds))
        mu = pm.Deterministic( "mu", pm.math.invprobit( (1/theta2) * pm.math.log(wind_speeds/theta1) ) )

        y_obs = pm.CustomDist('y_obs', PI_0, PI_1, mu, precision, logp=zero_one_beta_logp, observed=y)

        trace = pm.sample(3000, tune=1000, chains=4, cores=6, target_accept=0.99) # tuned samples are being discarded

        return zoib, trace
    
if __name__ == "__main__":

    theta1_theta2_mu_dict = {"bad": [220, 0.15], "medium": [270, 0.15], "good": [320, 0.15]}
    theta1_theta2_sd_dict = {"bad": [5, 0.015], "medium": [10, 0.03], "good": [10, 0.03]}
    building_type = 'bad'

    absolute_path = os.path.dirname(os.path.dirname(__file__))
    emp_data_file_path = f'{absolute_path}/assets/data/observations_{building_type}.csv'

    df = pd.read_csv(emp_data_file_path)
    v = df['x'].to_numpy()
    
    # norm_factor = v.max()
    norm_factor = 1

    v = np.array(v)/norm_factor

    theta_mus = theta1_theta2_mu_dict[building_type]
    theta_sds = theta1_theta2_sd_dict[building_type]
    theta1 = [theta_mus[0]/norm_factor, theta_sds[0]/norm_factor]
    theta2 = [theta_mus[1], theta_sds[1]]

    # synthetic data prep
    pi_0_params = get_mu_sd(lower_mu=0.01, prob_lower_mu=0.99, upper_mu=0.05, prob_upper_mu=0.01, building_type=building_type)
    pi_1_params = get_mu_sd(lower_mu=0.95, prob_lower_mu=0.01, upper_mu=0.99, prob_upper_mu=0.99, building_type=building_type)

    # We want theta3, theta4 to be defined in such a way that probability pi_0 is 0.99 when mu_y is 0.01 and 0.01 when the latter 0.05
    # For theta5, theta6, pi_1 is expected to be 0.01 when mu_y is 0.95 and 0.99 when mu_y is 0.99
    # sd is calculated to be 10% of the mean

    # True parameter values
    theta3 = [pi_0_params[0], abs(pi_0_params[0]/10)]
    theta4 = [pi_0_params[1], abs(pi_0_params[1]/10)]
    theta5 = [pi_1_params[0], abs(pi_1_params[0]/10)]
    theta6 = [pi_1_params[1], abs(pi_1_params[1]/10)]

    y = df['y'].to_numpy()

    # start = time.time()
    zoib, trace = do_MCMC([theta1, theta2, theta3, theta4, theta5, theta6], v, y)

    # summary statistics of the trace
    az.summary(trace)

    # plots of the trace
    az.plot_trace(data=trace, figsize=(30, 30), var_names=["theta1", "theta2", "theta3", "theta4", "theta5", "theta6", "precision"])

    az.plot_autocorr(trace, figsize=(30, 30), var_names=["theta1", "theta2", "theta3", "theta4", "theta5", "theta6", "precision"])

    # maybe not needed
    # az.plot_posterior(trace)

    pm.model_to_graphviz(zoib)

    # comments:
    # The effective sample size per chain is smaller than 100 for some parameters.  A higher number is needed for reliable rhat and ess computation.
    # end = time.time()