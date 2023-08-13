import pandas as pd
import numpy as np


def inv_logit(x):
    return np.exp(x)/(1 + np.exp(x))


trace_df = pd.read_csv(f"trace_data/trace_syn_bad.csv")
req_cols = ["theta1_chain0", "theta2_chain0", "theta3_chain0", "theta4_chain0", "theta5_chain0", "theta6_chain0", "precision_chain0"]
trace_df = trace_df.loc[:, req_cols]

lower_v = 5
upper_v = 500
v = np.arange(lower_v, upper_v + 1, step=1)
v_norm = v/upper_v

param_df = pd.DataFrame({"x_plot": pd.Series(dtype="str"), "iteration": pd.Series(dtype="str"), "zero": pd.Series(dtype="str"), "one": pd.Series(dtype="str"), "beta": pd.Series(dtype="str")})

v = list(v)
iteration = []
zero = []
one = []
beta = []
x_plot = []

for i in range(len(v)):
    wind_speed = v_norm[i]
    beta.extend(list(np.log(wind_speed * 1/trace_df["theta1_chain0"]) * 1/trace_df["theta2_chain0"]))
    zero.extend(list(inv_logit(trace_df["theta3_chain0"] + trace_df["theta4_chain0"] * wind_speed)))
    one.extend(list(inv_logit(trace_df["theta5_chain0"] + trace_df["theta6_chain0"] * wind_speed)))
    iteration.extend(list(np.arange(1, len(trace_df["theta1_chain0"]) + 1, step=1)))
    x_plot.extend([v[i]] * len(trace_df["theta1_chain0"]))

param_df["x_plot"] = x_plot
param_df["iteration"] = iteration
param_df["zero"] = zero
param_df["one"] = one
param_df["beta"] = beta

param_df.to_csv("trace_data/posteriors_bad_syn_our_model.csv")