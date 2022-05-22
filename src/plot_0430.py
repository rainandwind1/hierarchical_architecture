import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json
import copy
from smac.env import StarCraft2Env


plot_len = 200
tick_spacing = 500*1000
info_name = "info.json"
config_name = "config.json"
seed_num = 3

config_path_ls = [
    "pymarl-master/results/sacred/01/"+config_name,        
    "pymarl-master/results/sacred/02/"+config_name,
    "pymarl-master/results/sacred/03/"+config_name,
    "pymarl-master/results/sacred/95/"+config_name,        
    "pymarl-master/results/sacred/96/"+config_name,
    "pymarl-master/results/sacred/97/"+config_name,
    "pymarl-master/results/sacred/0101/"+config_name,        
    "pymarl-master/results/sacred/0102/"+config_name,
    "pymarl-master/results/sacred/0103/"+config_name, #
    "pymarl-master/results/sacred/111/"+config_name,        
    "pymarl-master/results/sacred/112/"+config_name,
    "pymarl-master/results/sacred/113/"+config_name,
    "pymarl-master/results/sacred/0107/"+config_name,        
    "pymarl-master/results/sacred/0108/"+config_name,
    "pymarl-master/results/sacred/0109/"+config_name,
    "pymarl-master/results/sacred/114/"+config_name,        
    "pymarl-master/results/sacred/115/"+config_name,
    "pymarl-master/results/sacred/0110/"+config_name,  #
    "pymarl-master/results/sacred/7/"+config_name,        
    "pymarl-master/results/sacred/8/"+config_name,
    "pymarl-master/results/sacred/15/"+config_name,
    "pymarl-master/results/sacred/23/"+config_name,        
    "pymarl-master/results/sacred/24/"+config_name,
    "pymarl-master/results/sacred/25/"+config_name,
    "pymarl-master/results/sacred/094/"+config_name,        
    "pymarl-master/results/sacred/095/"+config_name,
    "pymarl-master/results/sacred/096/"+config_name,         
    "pymarl-master/results/sacred/11/"+config_name,       
    "pymarl-master/results/sacred/12/"+config_name,
    "pymarl-master/results/sacred/17/"+config_name,
    "pymarl-master/results/sacred/20/"+config_name,        
    "pymarl-master/results/sacred/21/"+config_name,
    "pymarl-master/results/sacred/091/"+config_name,        
    "pymarl-master/results/sacred/092/"+config_name,
    "pymarl-master/results/sacred/093/"+config_name,
    "pymarl-master/results/sacred/22/"+config_name,         
    "pymarl-master/results/sacred/13/"+config_name,         
    "pymarl-master/results/sacred/14/"+config_name,
    "pymarl-master/results/sacred/16/"+config_name,
    "pymarl-master/results/sacred/26/"+config_name,        
    "pymarl-master/results/sacred/27/"+config_name,
    "pymarl-master/results/sacred/28/"+config_name,
    "pymarl-master/results/sacred/097/"+config_name,        
    "pymarl-master/results/sacred/098/"+config_name,
    "pymarl-master/results/sacred/099/"+config_name,         
]


info_path_ls = [
    "pymarl-master/results/sacred/01/"+info_name,        # 1c3s5z QMIX
    "pymarl-master/results/sacred/02/"+info_name,
    "pymarl-master/results/sacred/03/"+info_name,
    "pymarl-master/results/sacred/180/"+info_name,        #1c3s5z Graph_QMIX
    "pymarl-master/results/sacred/0150/"+info_name,
    "pymarl-master/results/sacred/0151/"+info_name,
    "pymarl-master/results/sacred/0145/"+info_name,        #1c3s5z multihead
    "pymarl-master/results/sacred/0148/"+info_name,
    "pymarl-master/results/sacred/0149/"+info_name,
    "pymarl-master/results/sacred/114/"+info_name,    # bane_vs_bane QMIX    
    "pymarl-master/results/sacred/115/"+info_name,
    "pymarl-master/results/sacred/0110/"+info_name,  
    "pymarl-master/results/sacred/0132/"+info_name,     # graphmix   
    "pymarl-master/results/sacred/0133/"+info_name,
    "pymarl-master/results/sacred/0147/"+info_name,
    "pymarl-master/results/sacred/0129/"+info_name,    # multihead    
    "pymarl-master/results/sacred/0130/"+info_name,
    "pymarl-master/results/sacred/0131/"+info_name, 
    # "pymarl-master/results/sacred/04/"+info_name,        # 3s5z_vs_3s6z QMIX
    # "pymarl-master/results/sacred/05/"+info_name,       
    # "pymarl-master/results/sacred/06/"+info_name,
    # "pymarl-master/results/sacred/98/"+info_name,        # 3s5z_vs_3s6z Graph_QMIX
    # "pymarl-master/results/sacred/99/"+info_name,
    # "pymarl-master/results/sacred/100/"+info_name, 
    "pymarl-master/results/sacred/7/"+info_name,          # MMM2 seed 01 QMIX
    "pymarl-master/results/sacred/8/"+info_name,          # MMM2 seed 02 QMIX
    "pymarl-master/results/sacred/15/"+info_name,         # MMM2 seed 03 QMIX
    "pymarl-master/results/sacred/167/"+info_name,         # MMM2 seed 01 Graph_QMIX
    "pymarl-master/results/sacred/168/"+info_name,         # MMM2 seed 02 Graph_QMIX
    "pymarl-master/results/sacred/169/"+info_name,         # MMM2 seed 03 Graph_QMIX
    "pymarl-master/results/sacred/198/"+info_name,         # multihead 
    "pymarl-master/results/sacred/199/"+info_name,       
    "pymarl-master/results/sacred/200/"+info_name,
    "pymarl-master/results/sacred/11/"+info_name,         # 2s3z seed 01 QMIX
    "pymarl-master/results/sacred/12/"+info_name,         # 2s3z seed 02 QMIX
    "pymarl-master/results/sacred/17/"+info_name,         # 2s3z seed 02 QMIX
    "pymarl-master/results/sacred/170/"+info_name,         # 2s3z seed 01 Graph_QMIX
    "pymarl-master/results/sacred/171/"+info_name,         # 2s3z seed 02 Graph_QMIX
    "pymarl-master/results/sacred/172/"+info_name,         # 2s3z seed 02 Graph_QMIX
    "pymarl-master/results/sacred/0139/"+info_name,        # multihead
    "pymarl-master/results/sacred/0140/"+info_name,
    "pymarl-master/results/sacred/0141/"+info_name,
    "pymarl-master/results/sacred/13/"+info_name,         # 8m   seed 01 QMIX
    "pymarl-master/results/sacred/14/"+info_name,         # 8m   seed 02 QMIX
    "pymarl-master/results/sacred/16/"+info_name,         # 8m   seed 03 QMIX
    "pymarl-master/results/sacred/173/"+info_name,         # 8m   seed 01 Graph_QMIX
    "pymarl-master/results/sacred/174/"+info_name,         # 8m   seed 02 Graph_QMIX
    "pymarl-master/results/sacred/175/"+info_name,         # 8m   seed 03 Graph_QMIX
    "pymarl-master/results/sacred/0142/"+info_name,        # multihead
    "pymarl-master/results/sacred/0143/"+info_name,
    "pymarl-master/results/sacred/0144/"+info_name,
]


def getdata():
    data = []
    alg_name = []
    scen_name = []
    for idx, config_path in enumerate(config_path_ls):
        with open(config_path, 'r') as f:
            config = json.load(f)
            if config["env_args"]["map_name"] not in scen_name:
                scen_name.append(config["env_args"]["map_name"])
            if config["name"] not in alg_name:
                alg_name.append(config["name"]) 

    info_count = 0
    for scen_i in range(len(scen_name)):
        data.append([])
        for alg_i in range(len(alg_name)):
            data[scen_i].append([])
            for seed_i in range(seed_num):
                with open(info_path_ls[info_count], 'r') as f:
                    info = json.load(f)
                    data[scen_i][alg_i].append(info["test_battle_won_mean"][:plot_len])
                info_count += 1
    return data, alg_name, scen_name

def plot_figure():
    data, alg_name, scen_name = getdata()
    xdata = [10000 * i for i in range(plot_len)]
    linestyle = ['-', '--', ':', '-.']
    color = ['r', 'g', 'b', 'k']
    sns.set_style(style = "whitegrid")
    for scen_i in range(len(scen_name)):
        for alg_i in range(len(alg_name)):
            for seed_i in range(len(data[scen_i][alg_i])):
                for idx in range(200):
                    if idx < 3:
                        data[scen_i][alg_i][seed_i][idx] = np.mean(data[scen_i][alg_i][seed_i][:idx+1])
                    else:
                        data[scen_i][alg_i][seed_i][idx] = np.mean(data[scen_i][alg_i][seed_i][idx-3:idx+1])

    for scen_i in range(len(scen_name)):
        fig = plt.figure()
        for alg_i in range(len(alg_name)):
            ax = sns.tsplot(time=xdata, data=data[scen_i][alg_i], color=color[alg_i], linestyle=linestyle[alg_i], condition=alg_name[alg_i])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.axes.xaxis.set_ticks([500*1000, 1000*1000 , 1500*1000, 2000*1000]) 
        ax.axes.set_xticklabels(['500.0k', '1.0m', '1.5m', '2.0m'])
        plt.xlabel("T", fontsize=15)
        plt.ylabel("Test Win Rate %", fontsize=15)
        plt.title('{}'.format(scen_name[scen_i]), fontsize=15)
        plt.savefig('pymarl-master/results/fig/{}.jpg'.format(scen_name[scen_i]))


if __name__ == "__main__":
    plot_figure()
    