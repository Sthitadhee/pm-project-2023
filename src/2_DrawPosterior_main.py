import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from helper.util import inv_logit
# Set the plot style
sns.set(style='ticks')
cmap = sns.color_palette("Set1")
# setting for printing for numpy (removes the e+00)
np.set_printoptions(suppress=True)
# Set display options to remove scientific notation
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
import os


def draw_graph(path, i=0, j=0, our_implementation = False):
    pos_db_bad = pd.read_csv(path)
    # v = pos_db_bad['x_plot'].to_numpy()
    # pi0 = pos_db_bad['zero'].to_numpy()
    # pi1 = pos_db_bad['one'].to_numpy()
    # mu = pos_db_bad['beta'].to_numpy()

    pos_db_trun_col = pos_db_bad[['x_plot', 'zero', 'one', 'beta']]
    pos_db_trun_col_row = pos_db_trun_col[~(pos_db_trun_col['x_plot'] > 325)]
    
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

    # Plot the interquartile range
    ax[i, j].fill_between(v, pi0_p10, pi0_p90, alpha=0.3, color='orange')  # Change to p10 and p90
    ax[i, j].plot(v, pi0_p25, linestyle='dotted', color='orange')  # Plot p25 as dotted line
    ax[i, j].plot(v, pi0_p75, linestyle='dotted', color='orange')  # Plot p75 as dotted line
    ax[i, j].plot(v, pi0_p50, linestyle='solid', color='orange')  # Plot p75 as dotted line

    # Plot the interquartile range
    ax[i, j].fill_between(v, pi1_p10, pi1_p90, alpha=0.3, color='green')  # Change to p10 and p90
    ax[i, j].plot(v, pi1_p25, linestyle='dotted', color='green', alpha=0.5)  # Plot p25 as dotted line
    ax[i, j].plot(v, pi1_p75, linestyle='dotted', color='green', alpha=0.5)  # Plot p75 as dotted line
    ax[i, j].plot(v, pi1_p50, linestyle='solid', color='green', alpha=0.5)  # Plot p75 as dotted line

    # Plot the interquartile range
    ax[i, j].fill_between(v, beta_p10, beta_p90, alpha=0.3, color='blue')  # Change to p10 and p90
    ax[i, j].plot(v, beta_p25, linestyle='dotted', color='blue', alpha=0.5)  # Plot p25 as dotted line
    ax[i, j].plot(v, beta_p75, linestyle='dotted', color='blue', alpha=0.5)  # Plot p75 as dotted line
    ax[i, j].plot(v, beta_p50, linestyle='solid', color='blue', alpha=0.5)  # Plot p75 as dotted line
    
    ax[i, j].set_xlabel('v, sustained wind speed (km/h)')
    ax[i, j].set_xlim(0, 325) # Adjust the x-axis range
    ax[i, j].set_ylabel('y, damage ratio [0-1]')
    ax[i, j].set_ylim(-0.05, 1.05)  # Adjust the y-axis range

    if b_type == 'bad':
        b_name = 'low-quality'
    elif b_type == 'medium':
        b_name = 'medium-quality'
    else: b_name = 'high-quality'

    if our_implementation:
        ax[i, j].set_title('P' + k_type[1:len(k_type)] + ' ' + b_name + ' (Genesis)')
    else:
        ax[i, j].set_title('P' + k_type[1:len(k_type)] + ' ' + b_name)

if __name__ == '__main__':
    building_type = ['bad', 'medium', 'good']
    knowledge_type_author = ['priors', 'posteriors']
    knowledge_type_implementation = ['posteriors']
    absolute_path = os.path.dirname(os.path.dirname(__file__))
    # use author data True to replicate the author's graphings
    # use author data False to show our implementation of the author's model for graphing
    author_data = True # change this to check
    implementation_data = True # change this to check

    for i, k_type in enumerate(knowledge_type_author):
        for j, b_type in enumerate(building_type):
            path = f'{absolute_path}/assets/{k_type}/{k_type}_{b_type}.csv'
            author_data = os.path.isfile(path) # change this to check
    for i, k_type in enumerate(knowledge_type_implementation):
        for j, b_type in enumerate(building_type):
            path = f'{absolute_path}/assets/{k_type}_genesis/{k_type}_{b_type}_implementation.csv'
            implementation_data = os.path.isfile(path) # change this to check

    image_row = 0
    # Set the figure and axis objects
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 6))

    for i, k_type in enumerate(knowledge_type_author):
        for j, b_type in enumerate(building_type):
            if author_data == True:
                path = f'{absolute_path}/assets/{k_type}/{k_type}_{b_type}.csv'
                draw_graph(path, i, j)
                image_row = i
            

    for i, k_type in enumerate(knowledge_type_implementation):
        for j, b_type in enumerate(building_type):
            if implementation_data == True:
                if k_type == 'priors': continue
                path = f'{absolute_path}/assets/{k_type}_genesis/{k_type}_{b_type}_implementation.csv'
                draw_graph(path, i+image_row+1, j, True)

    fig.legend(
        ['π0 P10-P90', 'P25', 'P75', 'P50', 'π1 P10-P90', 'P25', 'P75', 'P50', 'μ P10-P90', 'P25', 'P75', 'P50'], 
        loc='outside right upper', 
        bbox_to_anchor=(1, -0.02),
    )
    plt.tight_layout(pad=1)
    plt.show()
