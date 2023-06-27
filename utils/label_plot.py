import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np
import seaborn as sns

plt.style.use("dark_background")

label_dict = {1: 'Still', 2: 'Walking', 3: 'Run', 4: 'Bike', 5: 'Car', 6: 'Bus', 7: 'Train', 8: 'Subway'}
cmap = mpl.cm.get_cmap('viridis', 8)
label_cols = cmap(np.linspace(0, 1, 8))
label_cols = [mpl.colors.to_hex(i) for i in label_cols]

def plot_label_each(df, col_name, this_label):
    label_index = np.where(df.label == this_label)[0]
    p = plt.scatter(df.index[label_index], df[col_name][label_index], c = label_cols[this_label - 1], label = label_dict[this_label])
    return p

def plot_label(df, col_name):
    plt.figure(figsize = [12, 8])
    for this_label in np.unique(list(df.label)):
        plot_label_each(df, col_name, this_label)
    plt.legend(loc = 'best')

def box_label(df, col_name, use_log = False, violin = False):
    plt.figure(figsize = [12, 8])
    y = np.log(df[col_name]+1) if use_log else df[col_name]
    if violin:
        sns.violinplot(x = df['label'], y = y)
    else:
        sns.boxplot(x = df['label'], y = y)
    plt.xticks(range(8), list(label_dict.values()))