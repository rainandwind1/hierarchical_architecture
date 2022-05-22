from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import random
import pandas as pd
from sklearn.cluster import SpectralClustering, KMeans

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0
        self.t_macro = 0

        self.goal_t = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        # for test goal random goal candidate
        self.goal_box = [[np.ones(self.env.get_goal_size()) for _ in range(self.args.goal_num)] for _ in range(self.env.n_agents)]

    def setup(self, goal_scheme, scheme, groups, preprocess, goal_preprocess, mac, goal_mac):
        # micro trans
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
                                 
        # macro goal trans                         
        self.new_goal_batch = partial(EpisodeBatch, goal_scheme, groups, self.batch_size, int(self.episode_limit / self.args.horizon) + 1,
                                 preprocess=goal_preprocess, device=self.args.device)
        
        # controller
        self.mac = mac
        self.goal_mac = goal_mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.goal_batch = self.new_goal_batch()
        self.env.reset()
        self.t = 0
        self.goal_t = 0

    def goal_id_to_goal(self, goal_ids):
        # goals = []
        # for idx, goal_id in enumerate(goal_ids):
        #     goals.append(self.goal_box[int(goal_id)])
        return [np.ones(self.env.get_goal_size()) for _ in range(self.env.n_agents)]

    def get_avail_goal_ids(self):
        avail_goal_ids_ls = []
        self.n_goals = self.args.goal_num
        for i in range(self.env.n_agents):
            avail_goal_ids_ls.append([1] * self.n_goals)
        return avail_goal_ids_ls

    def cal_intrinsic_reward(self, goal_feats, goals):
        intrinsic_reward_ls = [[]]
        for goal_feat, goal in zip(goal_feats, goals):
            intrinsic_reward_ls[0].append(-np.linalg.norm(goal_feat - goal))
        return intrinsic_reward_ls

    def goal_exploration(self, logger, test_mode=False):
        logger.console_logger.info("Begin exploration goals: Random walk!")
        if test_mode:
            pass
        
        t_random = 0
        epi_random = 0.
        end = False
        goal_obs_feats_buffer = [[] for _ in range(self.env.n_agents)]
        good_goal_obs_feats_buffer = [[] for _ in range(self.env.n_agents)]
        env_info = self.get_env_info()
        while t_random < self.args.random_walk:
            epi_random += 1
            self.env.reset()
            end = False
            while not end:  
                # random action choose and execute
                actions = []
                for idx in range(self.env.n_agents):
                    avail_mask = self.env.get_avail_agent_actions(idx)
                    candidate_action = range(env_info["n_actions"])
                    candidate_action = [i for i in candidate_action if avail_mask[i] > 0]
                    action = random.sample(candidate_action, 1)[0]
                    actions.append(action)
                
                reward, terminated, info = self.env.step(actions)

                # s_next
                goal_obs_feats_ls = self.env.get_goal_feats()
                for agent_i in range(self.env.n_agents):
                    if reward <= 0:
                        goal_obs_feats_buffer[agent_i].append((goal_obs_feats_ls[agent_i], reward))
                    else:
                        good_goal_obs_feats_buffer[agent_i].append((goal_obs_feats_ls[agent_i], reward))

                # update counter
                t_random += 1
                end = terminated

                if (t_random + 1) % 10000 == 0:
                    logger.console_logger.info("{} Step execute!".format(t_random + 1))
        
        logger.console_logger.info("Update goal box!")
        self.update_goal_box(goal_obs_feats_buffer, good_goal_obs_feats_buffer)
        logger.console_logger.info("End exploration goals: Save goals to excel!")
    

    def update_goal_box(self, goal_obs_feats_buffer, good_goal_obs_feats_buffer):
        # kmeans + anomaly detection

        # anomaly detection
        for idx, goal_obs_trans in enumerate(good_goal_obs_feats_buffer):
            goal_obs_trans = sorted(goal_obs_trans, key = lambda item:item[1], reverse=True)
            goal_obs_data = [trans[0] for trans in goal_obs_trans if trans[1] > 0]
            if len(goal_obs_data) > self.args.goal_num / 2:
                self.goal_box[idx] = []
                count = 0
                # 重复检测
                while len(self.goal_box[idx]) < self.args.goal_num / 2:
                    if list(goal_obs_data[count]) not in self.goal_box[idx]:
                        self.goal_box[idx].append(list(goal_obs_data[count]))
                    count += 1
            else:
                self.goal_box[idx] = []
                count = 0
                # 重复检测
                while count < len(goal_obs_data):
                    if list(goal_obs_data[count]) not in self.goal_box[idx]:
                        self.goal_box[idx].append(list(goal_obs_data[count]))
                    count += 1

        # kmeans
        for idx, goal_obs_trans in enumerate(goal_obs_feats_buffer):
            goal_obs_data = [trans[0] for trans in goal_obs_trans]
            s = KMeans(n_clusters=self.args.goal_num, random_state=0).fit(goal_obs_data)
            goal_obs_core = s.cluster_centers_
            #goal_obs_core = self.kmeans(np.stack(goal_obs_data), self.args.goal_num)
            for i in range(goal_obs_core.shape[0]):
                if list(goal_obs_core[i]) not in self.goal_box[idx]:
                    self.goal_box[idx].append(list(goal_obs_core[i]))
            # 限长
            self.goal_box[idx] = self.goal_box[idx][:self.args.goal_num]

        self.save_goals()

    def save_goals(self):
        writer = pd.ExcelWriter('./goal.xlsx')		# 写入Excel文件
        
        for agent_i, agent_goal_ls in enumerate(self.goal_box):
            data = pd.DataFrame(np.array(agent_goal_ls))
            data.to_excel(writer, 'page_' + '{}'.format(agent_i), float_format='%.5f')		# ‘page_1’是写入excel的sheet名
        
        writer.save()
        writer.close()
                        
    
    def kmeans(self, ds, k):
        """ k-means聚类算法
        k       - 指定分簇数量
        ds      - ndarray(m, n)，m个样本的数据集，每个样本n个属性值
        """
        
        m, n = ds.shape # m：样本数量，n：每个样本的属性值个数
        result = np.empty(m, dtype=np.int) # m个样本的聚类结果
        cores = ds[np.random.choice(np.arange(m), k, replace=False)] # 从m个数据样本中不重复地随机选择k个样本作为质心
        count = 0
        while True: # 迭代计算
            d = np.square(np.repeat(ds, k, axis=0).reshape(m, k, n) - cores)
            distance = np.sqrt(np.sum(d, axis=2)) # ndarray(m, k)，每个样本距离k个质心的距离，共有m行
            index_min = np.argmin(distance, axis=1) # 每个样本距离最近的质心索引序号
            
            if (index_min == result).all(): # 如果样本聚类没有改变
                return result, cores # 则返回聚类结果和质心数据
            
            result[:] = index_min # 重新分类
            for i in range(k): # 遍历质心集
                items = ds[result==i] # 找出对应当前质心的子样本集
                cores[i] = np.mean(items, axis=0) # 以子样本集的均值作为当前质心的位置


    def run(self, test_mode=False):
        self.reset()

        # for debug
        # state_debug = self.env.get_state()
        # obs_debug = self.env.get_obs()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        
        # goal
        self.goal_mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            goal_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.get_avail_goal_ids()],
                "obs": [self.env.get_obs()],
                "goal_obs": [self.env.get_goal_feats()],
                "adjacency_matrix":[self.env.get_adjacency_matrix()]
            }
            self.goal_batch.update(goal_transition_data, ts=self.goal_t)
            
            goal_interval = 0       # macro goal achieved flag next choose new goal for micro policy
            goal_return = 0.
            goal_ids = self.goal_mac.select_actions(self.goal_batch, t_ep=self.goal_t, t_env=self.t_macro, test_mode=test_mode)
            goals = self.goal_id_to_goal(goal_ids[0])                          # *********************************************************** 待调
            

            while goal_interval < self.args.horizon and not terminated:
                # obs divided into 4 items: move feats、enemy feats、ally feats、own feats
                agent_obs_feats = self.env.get_obs_feats()
                obs_ls = self.env.get_obs()
                micro_obs = [np.concatenate([agent_obs, goal]) for agent_obs, goal in zip(obs_ls, goals)]

                pre_transition_data = {
                    "state": [self.env.get_state()],
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [micro_obs],
                    "adjacency_matrix":[self.env.get_adjacency_matrix()],
                    "move_feats":[agent_obs_feats[0]],
                    "enemy_feats":[agent_obs_feats[1]],
                    "ally_feats":[agent_obs_feats[2]],
                    "own_feats":[agent_obs_feats[3]]
                }

                self.batch.update(pre_transition_data, ts=self.t)

                # Pass the entire batch of experiences up till now to the agents
                # Receive the actions for each agent at this timestep in a batch of size 1
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

                # print(actions.shape)
                reward, terminated, env_info = self.env.step(actions[0])
                intrinsic_reward = self.cal_intrinsic_reward(self.env.get_goal_feats(), goals)                      ## ************** 待调
                episode_return += reward
                goal_return += reward

                post_transition_data = {
                    "actions": actions,
                    "reward": intrinsic_reward,
                    "terminated": [(terminated != env_info.get("episode_limit", False),)]
                }

                self.batch.update(post_transition_data, ts=self.t)

                self.t += 1
                goal_interval += 1
            
            goal_post_transition_data = {
                "actions": goal_ids,
                "reward": [(goal_return,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)]
            }
            self.goal_batch.update(goal_post_transition_data, ts=self.goal_t)
            self.goal_t += 1

        agent_obs_feats = self.env.get_obs_feats()
        micro_obs = [np.concatenate([agent_obs, goal]) for agent_obs, goal in zip(obs_ls, goals)]
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [micro_obs],
            "adjacency_matrix":[self.env.get_adjacency_matrix()],
            "move_feats":[agent_obs_feats[0]],
            "enemy_feats":[agent_obs_feats[1]],
            "ally_feats":[agent_obs_feats[2]],
            "own_feats":[agent_obs_feats[3]]
        }
        self.batch.update(last_data, ts=self.t)

        last_goal_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.get_avail_goal_ids()],
            "obs": [self.env.get_obs()],
            "goal_obs": [self.env.get_goal_feats()],
            "adjacency_matrix":[self.env.get_adjacency_matrix()]
        }
        self.goal_batch.update(last_goal_data, ts=self.goal_t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        # Select goals in the last stored state
        goal_ids = self.goal_mac.select_actions(self.goal_batch, t_ep=self.goal_t, t_env=self.t_macro, test_mode=test_mode)
        self.batch.update({"actions": goal_ids}, ts=self.goal_t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.t_macro += self.goal_t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, self.goal_batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
