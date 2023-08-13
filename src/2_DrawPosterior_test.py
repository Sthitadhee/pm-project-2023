import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from helper.util import inv_logit
# Set the plot style
sns.set(style='ticks')
cmap = sns.color_palette("Set1")
# setting for printing for numpy (removes the e+00)
np.set_printoptions(suppress=True)
# Set display options to remove scientific notation
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))

# posterior_db = pd.read_csv("../assets/posteriors/posterior_thetas_bad.csv")
# wind_v = pd.read_csv("./observations_bad.csv")
# v = wind_v['x'].to_numpy(dtype=np.int32)
# y = wind_v['y'].to_numpy()
# v = np.arange(0, 310, 5)

# # drop the first column
# posterior_db_dropped = posterior_db.iloc[:, 1:]

# # sort along the columns phi, theta1-6
# posterior_db_sorted = posterior_db_dropped.sort_index(axis=1)
# theta_3 = posterior_db_sorted['theta_3'].to_numpy()
# theta_4 = posterior_db_sorted['theta_4'].to_numpy()

# theta3_p50 = np.percentile(theta_3, 50)
# theta3_mean = np.mean(theta_3)
# theta3_std = np.std(theta_3)
# theta3_sample = np.random.normal(theta3_mean, theta3_std, len(v))
# # theta3_p10
# # theta3_p25
# # theta3_p75
# # theta3_p90

# theta4_p50 = np.percentile(theta_4, 50)
# theta4_mean = np.mean(theta_4/100)
# theta4_std = np.std(theta_4/100)
# theta4_sample = np.random.normal(-0.55, 0.05, len(v))
# # theta4_p10
# # theta4_p25
# # theta4_p75
# # theta4_p90

# # theta5_mean
# # theta5_p10
# # theta5_p25
# # theta5_p75
# # theta5_p90

# # theta6_mean
# # theta6_p10
# # theta6_p25
# # theta6_p75
# # theta6_p90
# # plt.scatter(v, y, color='black', label='Observations', alpha=0.8, s=5)
# # plt.xlim(0, 310)
# # pi0_p50 = inv_logit(theta3_p50 + theta4_p50 * v)
# pi0_p50 = inv_logit(theta3_sample + theta4_sample * v)
# # pi0_p10 = inv_logit(theta3_mu + theta4_mu * np.mean(v))
# # pi0_p90 = inv_logit(theta3_mu + theta4_mu * np.mean(v))
# # pi0_p25 = inv_logit(theta3_mu + theta4_mu * np.mean(v))
# # pi0_p75 = inv_logit(theta3_mu + theta4_mu * np.mean(v))

# # pi1_p50 = inv_logit(theta5_mu + theta6_mu * np.mean(v))
# # pi1_p10 = inv_logit(theta5_mu + theta6_mu * np.mean(v))
# # pi1_p90 = inv_logit(theta5_mu + theta6_mu * np.mean(v))
# # pi1_p25 = inv_logit(theta5_mu + theta6_mu * np.mean(v))
# # pi1_p75 = inv_logit(theta5_mu + theta6_mu * np.mean(v))


# # Set the figure and axis objects
# fig, ax = plt.subplots()
# ax.plot(v, pi0_p50, linestyle='solid', color=cmap[0])  # Plot p75 as dotted line
# plt.show()



# # Generate example data
# x = np.arange(0, 310) # Predictor variable
# y = np.random.rand(len(x))  # Response variable

# # Calculate percentiles
# p10 = np.percentile(y, 10)
# p25 = np.percentile(y, 25)
# p50 = np.percentile(y, 50)
# p75 = np.percentile(y, 75)
# p90 = np.percentile(y, 90)

# # Create a DataFrame for plotting
# data = pd.DataFrame({'x': x, 'y': y})

# # Set the figure and axis objects
# fig, ax = plt.subplots()

# # Plot the interquartile range
# ax.fill_between(x, p10, p90, alpha=0.3, color=cmap[0])  # Change to p10 and p90
# ax.plot(x, [p25] * len(x), linestyle='dotted', color=cmap[0])  # Plot p25 as dotted line
# ax.plot(x, [p75] * len(x), linestyle='dotted', color=cmap[0])  # Plot p75 as dotted line
# ax.plot(x, [p50] * len(x), linestyle='solid', color=cmap[0])  # Plot p75 as dotted line
# plt.scatter(x, y, color='black', label='Observations', alpha=0.8, s=5)

# # Add gridlines
# ax.grid(True, linestyle='dotted', alpha=0.5)

# # Set the background color
# ax.set_facecolor('white')

# # Add labels and title
# ax.set_xlabel('v, sustained wind speed (km/h)')
# ax.set_xlim(0, 320) # Adjust the x-axis range
# ax.set_ylabel('y, damage ratio [0-1]')
# ax.set_ylim(-0.01, 1)  # Adjust the y-axis range
# ax.set_title('Regression Graph with Interquartile Range')

# Show the plot
# plt.show()

building_type = [('bad', 'orange'), ('medium', 'c'), ('good', 'b')]

def draw_graph(file_name=f"../assets/posteriors/posteriors_bad.csv"):
    pos_db_bad = pd.read_csv(file_name)
    # v = pos_db_bad['x_plot'].to_numpy()
    # pi0 = pos_db_bad['zero'].to_numpy()
    # pi1 = pos_db_bad['one'].to_numpy()
    # mu = pos_db_bad['beta'].to_numpy()

    pos_db_trun_col = pos_db_bad[['x_plot', 'zero', 'one', 'beta']]
    pos_db_trun_col_row = pos_db_trun_col[~(pos_db_trun_col['x_plot'] > 310)]
    
    # get winds speeds range
    v = np.unique(pos_db_trun_col_row['x_plot'].to_numpy())

    df_with_percentiles = pos_db_trun_col_row.groupby('x_plot').quantile([.10, .25, .5, .75, .90])
    pi0_p10 = df_with_percentiles['zero'].xs(0.1, level=1).to_numpy()
    pi0_p25 = df_with_percentiles['zero'].xs(0.25, level=1).to_numpy()
    pi0_p50 = df_with_percentiles['zero'].xs(0.5, level=1).to_numpy()
    pi0_p75 = df_with_percentiles['zero'].xs(0.75, level=1).to_numpy()
    pi0_p90 = df_with_percentiles['zero'].xs(0.90, level=1).to_numpy()

    pi1_p10 = df_with_percentiles['one'].xs(0.1, level=1).to_numpy()
    pi1_p25 = df_with_percentiles['one'].xs(0.25, level=1).to_numpy()
    pi1_p50 = df_with_percentiles['one'].xs(0.5, level=1).to_numpy()
    pi1_p75 = df_with_percentiles['one'].xs(0.75, level=1).to_numpy()
    pi1_p90 = df_with_percentiles['one'].xs(0.9, level=1).to_numpy()
    
    beta_p10 = df_with_percentiles['beta'].xs(0.1, level=1).to_numpy()
    beta_p25 = df_with_percentiles['beta'].xs(0.25, level=1).to_numpy()
    beta_p50 = df_with_percentiles['beta'].xs(0.5, level=1).to_numpy()
    beta_p75 = df_with_percentiles['beta'].xs(0.75, level=1).to_numpy()
    beta_p90 = df_with_percentiles['beta'].xs(0.90, level=1).to_numpy()

    # SLOW
    # pi0_p10 = pos_db_trun.groupby('x_plot').quantile(.10)['zero'].to_numpy()
    # pi0_p25 = pos_db_trun.groupby('x_plot').quantile(.25)['zero'].to_numpy()
    # pi0_p50 = pos_db_trun.groupby('x_plot').quantile(.50)['zero'].to_numpy()
    # pi0_p75 = pos_db_trun.groupby('x_plot').quantile(.75)['zero'].to_numpy()
    # pi0_p90 = pos_db_trun.groupby('x_plot').quantile(.90)['zero'].to_numpy()
    # v = np.unique(pos_db_trun['x_plot'].to_numpy())

    # pi1_p10 = pos_db_trun.groupby('x_plot').quantile(.10)['one'].to_numpy()
    # pi1_p25 = pos_db_trun.groupby('x_plot').quantile(.25)['one'].to_numpy()
    # pi1_p50 = pos_db_trun.groupby('x_plot').quantile(.50)['one'].to_numpy()
    # pi1_p75 = pos_db_trun.groupby('x_plot').quantile(.75)['one'].to_numpy()
    # pi1_p90 = pos_db_trun.groupby('x_plot').quantile(.90)['one'].to_numpy()
    
    # beta_p10 = pos_db_trun.groupby('x_plot').quantile(.10)['beta'].to_numpy()
    # beta_p25 = pos_db_trun.groupby('x_plot').quantile(.25)['beta'].to_numpy()
    # beta_p50 = pos_db_trun.groupby('x_plot').quantile(.50)['beta'].to_numpy()
    # beta_p75 = pos_db_trun.groupby('x_plot').quantile(.75)['beta'].to_numpy()
    # beta_p90 = pos_db_trun.groupby('x_plot').quantile(.90)['beta'].to_numpy()



    # Set the figure and axis objects
    fig, ax = plt.subplots()

        
    # Plot the interquartile range
    ax.fill_between(v, pi0_p10, pi0_p90, alpha=0.3, color='orange')  # Change to p10 and p90
    ax.plot(v, pi0_p25, linestyle='dotted', color='orange')  # Plot p25 as dotted line
    ax.plot(v, pi0_p75, linestyle='dotted', color='orange')  # Plot p75 as dotted line
    ax.plot(v, pi0_p50, linestyle='solid', color='orange')  # Plot p75 as dotted line

    # Plot the interquartile range
    ax.fill_between(v, pi1_p10, pi1_p90, alpha=0.3, color='green')  # Change to p10 and p90
    ax.plot(v, pi1_p25, linestyle='dotted', color='green')  # Plot p25 as dotted line
    ax.plot(v, pi1_p75, linestyle='dotted', color='green')  # Plot p75 as dotted line
    ax.plot(v, pi1_p50, linestyle='solid', color='green')  # Plot p75 as dotted line

    # Plot the interquartile range
    ax.fill_between(v, beta_p10, beta_p90, alpha=0.3, color='blue')  # Change to p10 and p90
    ax.plot(v, beta_p25, linestyle='dotted', color='blue')  # Plot p25 as dotted line
    ax.plot(v, beta_p75, linestyle='dotted', color='blue')  # Plot p75 as dotted line
    ax.plot(v, beta_p50, linestyle='solid', color='blue')  # Plot p75 as dotted line
    plt.show()


# testing purpose

# Read data from the file
absolute_path = os.path.dirname(os.path.dirname(__file__))
file_name = f"{absolute_path}/results/trace_plots_bad_empirical.csv"
draw_graph(file_name)