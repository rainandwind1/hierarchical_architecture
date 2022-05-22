from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json
import copy
from smac.env import StarCraft2Env


plot_len = 201
tick_spacing = 500*1000
info_name = "info.json"
config_name = "config.json"
seed_num = 3

# path_number_ls = [204, 208, 212, 205, 209, 213, 281,282,283]      # 3s 4z
# path_number_ls = [222, 223, 224, 225, 228, 229,284,285,286]           # 2c 64zg
# path_number_ls = [11, 1089, 1091, 1035] # 492, 498]   # 2s3z
# path_number_ls = [7, 8, 15, 167, 168, 169, 290, 288, 291]              # mmm2
# path_number_ls = [999, 1104]# [114, 999, 1103, 1104]   # bane   
# path_number_ls = [1200, 1201, 1202, 1110, 1111, 1112] # 5m 6m
# path_number_ls = [1206, 1207, 1208, 1203, 1204, 1205]  # 8m 9m

# sparse reward
#  path_number_ls = [415, 416, 417]    # 2s3z
# path_number_ls = [411, 413, 414]



config_path_ls = []
info_path_ls = []
for num in path_number_ls:
    config_path_ls.append("pymarl-master/results/sacred/{}/".format(num) + config_name)
    info_path_ls.append("pymarl-master/results/sacred/{}/".format(num) + info_name)

# for num_ls in path_number_ls:
#     for num in num_ls:
#         config_path_ls.append("pymarl-master/results/sacred/{}/".format(num) + config_name)
#         info_path_ls.append("pymarl-master/results/sacred/{}/".format(num) + info_name)


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

def plot_map_info(map_name, n_agents):
    for i in range(n_agents):
        data = np.load('pymarl-master/{}'.format(map_name) + '_{}.npy'.format(i), allow_pickle=True)
        
        print(data.shape)
        fig, ax = plt.subplots()
        
        colors = ['k','y']
        label = ['move', 'attack']
        marker = ['s','*']
        for c_id, color in enumerate(colors):
            need_idx = np.where(data[:,3]==c_id)[0]
            ax.scatter(data[need_idx,0],data[need_idx,1], c=color, marker=marker[c_id], label=label[c_id], alpha = 0.6)
        
        # plt.xlim((0, 50))
        # plt.ylim((0, 50))
        plt.title("{} - agent_{}".format(map_name, i))
        legend = ax.legend()
        plt.savefig("pymarl-master/results/hrl_map_fig/{}_{}.png".format(map_name, i))
        print("save success!")


def plot_figure():
    data, alg_name, scen_name = getdata()
    xdata = [10000 * i for i in range(plot_len)]
    linestyle = ['-', '--', ':', '-.']
    color = ['r', 'g', 'b', 'k']
    sns.set_style(style = "whitegrid")
    for scen_i in range(len(scen_name)):
        for alg_i in range(len(alg_name)):
            for seed_i in range(len(data[scen_i][alg_i])):
                for idx in range(len(data[scen_i][alg_i][seed_i])):
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
    # plot_map_info("2s3z", 5)
    



