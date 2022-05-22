import enum
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

        # total
        self.t_env = 0
        self.t_macro = 0
        self.t_move = 0
        self.t_action = 0

        # epi count
        self.macro_t = 0
        self.move_t = 0
        self.action_t = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        # for test goal random goal candidate
        self.goal_box = [[np.ones(self.env.get_goal_size()) for _ in range(self.args.goal_num)] for _ in range(self.env.n_agents)]

    #(macro_scheme=macro_scheme, move_scheme=move_scheme, action_scheme=action_scheme, groups=groups, move_preprocess=move_preprocess, action_preprocess=action_preprocess, macro_preprocess=macro_preprocess, move_mac=move_mac, action_mac=action_mac, macro_mac=macro_mac)
    def setup(self, macro_scheme, move_scheme, action_scheme, groups, move_preprocess, action_preprocess, macro_preprocess, move_mac, action_mac, macro_mac):
        # micro trans
        self.new_move_batch = partial(EpisodeBatch, move_scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=move_preprocess, device=self.args.device)

        # micro trans
        self.new_action_batch = partial(EpisodeBatch, action_scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=action_preprocess, device=self.args.device)
                                 
        # macro goal trans                         
        self.new_macro_batch = partial(EpisodeBatch, macro_scheme, groups, self.batch_size, int(self.episode_limit / self.args.min_horizon) + 1,
                                 preprocess=macro_preprocess, device=self.args.device)
        
        # controller
        self.map_data = [[] for _ in range(self.env.n_agents)]
        self.move_mac = move_mac
        self.action_mac = action_mac
        self.macro_mac = macro_mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.move_batch = self.new_move_batch()
        self.action_batch = self.new_action_batch()
        self.macro_batch = self.new_macro_batch()
        self.env.reset()

        # episode step counter
        self.t = 0
        self.macro_t = 0
        self.move_t = 0
        self.action_t = 0

    def goal_id_to_goal(self, goal_ids):
        goals = []
        for idx, goal_id in enumerate(goal_ids):
            goals.append(self.goal_box[idx][int(goal_id)])
        return goals

    def get_avail_macro_ids(self, all_actions):
        avail_macro_ids_ls = []
        for i in range(self.env.n_agents):
            agent_avail_action_ls = all_actions[i]
            res = [0, 0]
            if 1 in agent_avail_action_ls[:self.args.a_move_size]:
                res[0] = 1
            if 1 in agent_avail_action_ls[self.args.a_move_size:]:
                res[1] = 1
            avail_macro_ids_ls.append(res)
            # print(avail_macro_ids_ls)
        return avail_macro_ids_ls

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
            if len(goal_obs_data) > self.args.goal_num:
                self.goal_box[idx] = []
                count = 0
                # 重复检测
                while len(self.goal_box[idx]) < self.args.goal_num:
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
        
        if len(self.goal_box[idx]) < self.args.goal_num:
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

    def get_avail_micro_actions(self, all_avail_actions, macro_ids):
        avail_actions = []
        for idx, all_avail_action in enumerate(all_avail_actions):
            if int(macro_ids[idx]) == 0:
                avail_actions.append(all_avail_action[:self.args.a_move_size] + [0] * (len(all_avail_action) - self.args.a_move_size))
            elif int(macro_ids[idx]) == 1:
                avail_actions.append([0] * self.args.a_move_size + all_avail_action[self.args.a_move_size:])
        return avail_actions

    def get_avail_move_a_actions(self, all_avail_actions):
        avail_move_ids = []
        avail_a_ids = []
        for idx, all_avail_action in enumerate(all_avail_actions):
            avail_move_ids.append(all_avail_action[:self.args.a_move_size])
            avail_a_ids.append(all_avail_action[self.args.a_move_size:])
            if 1 not in avail_a_ids[-1]:
                avail_a_ids[-1] += [1]     # no op added for action policy action selection
            else:
                avail_a_ids[-1] += [0]
        return avail_move_ids, avail_a_ids

    def get_actions(self, move_actions, a_actions, macro_ids):
        actions = []
        for idx, macro_id in enumerate(macro_ids):
            if int(macro_id) == 0:
                actions.append(move_actions[idx])
            elif int(macro_id) == 1:
                actions.append(a_actions[idx] + self.args.a_move_size)
        return actions

    def decomposing_reward(self, reward, macro_ids):
        count = 0
        for action in macro_ids:
            if int(action) == 0:
                count += 1
        reward_move = (count / len(macro_ids)) * reward
        reward_a = reward - reward_move
    
        return reward_move, reward_a
        



    def run(self, test_mode=False):
        self.reset()

        # for debug
        # state_debug = self.env.get_state()
        # obs_debug = self.env.get_obs()

        terminated = False
        episode_return = 0

        # micro mac initialize hidden weight
        self.move_mac.init_hidden(batch_size=self.batch_size)
        self.action_mac.init_hidden(batch_size=self.batch_size)
        
        # goal
        self.macro_mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            all_avail_actions = self.env.get_avail_actions()
            macro_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.get_avail_macro_ids(all_avail_actions)],
                "obs": [self.env.get_obs()],
                "goal_obs": [self.env.get_goal_feats()],
                "adjacency_matrix":[self.env.get_adjacency_matrix()]
            }
            avail_actions_test = self.get_avail_macro_ids(all_avail_actions)
            self.macro_batch.update(macro_transition_data, ts=self.macro_t)
            
            macro_interval = 0       # macro goal achieved flag next choose new goal for micro policy
            macro_return = 0.
            macro_ids = self.macro_mac.select_actions(self.macro_batch, t_ep=self.macro_t, t_env=self.t_macro, test_mode=test_mode)
            

            # if test_mode: # for show macro policy
            #     pos_agents = self.env.get_pos_agents()
            #     for idx, agent_info in enumerate(pos_agents):
            #         agent_info.append(macro_ids[0][idx])
            #         self.map_data[idx].append(agent_info)

            while macro_interval < self.args.min_horizon and not terminated:
                # obs divided into 4 items: move feats、enemy feats、ally feats、own feats
                agent_obs_feats = self.env.get_obs_feats()
                
                # all_avail_actions = self.env.get_avail_actions()
                avail_move_actions_ls, avail_a_actions_ls = self.get_avail_move_a_actions(all_avail_actions)            # 待调

                pre_move_transition_data = {
                    "state": [self.env.get_state()],
                    "avail_actions": [avail_move_actions_ls],
                    "obs": [self.env.get_obs()],
                    "adjacency_matrix":[self.env.get_adjacency_matrix()],
                    "move_feats":[agent_obs_feats[0]],
                    "enemy_feats":[agent_obs_feats[1]],
                    "ally_feats":[agent_obs_feats[2]],
                    "own_feats":[agent_obs_feats[3]]
                }

                pre_a_transition_data = {
                    "state": [self.env.get_state()],
                    "avail_actions": [avail_a_actions_ls],
                    "obs": [self.env.get_obs()],
                    "adjacency_matrix":[self.env.get_adjacency_matrix()],
                    "move_feats":[agent_obs_feats[0]],
                    "enemy_feats":[agent_obs_feats[1]],
                    "ally_feats":[agent_obs_feats[2]],
                    "own_feats":[agent_obs_feats[3]]
                }

                self.move_batch.update(pre_move_transition_data, ts=self.t)
                self.action_batch.update(pre_a_transition_data, ts=self.t)

                # Pass the entire batch of experiences up till now to the agents
                # Receive the actions for each agent at this timestep in a batch of size 1
                move_actions = self.move_mac.select_actions(self.move_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                a_actions = self.action_mac.select_actions(self.action_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                actions = self.get_actions(move_actions[0], a_actions[0], macro_ids[0])

                # print(actions.shape)
                reward, terminated, env_info = self.env.step(actions)
            
                episode_return += reward
                macro_return += reward
                reward_move, reward_a = reward, reward

                post_a_transition_data = {
                    "actions": a_actions,
                    "reward": [(reward_a,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)]
                }

                self.action_batch.update(post_a_transition_data, ts=self.t)

                post_move_transition_data = {
                    "actions": move_actions,
                    "reward": [(reward_move,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)]
                }

                self.move_batch.update(post_move_transition_data, ts=self.t)

                self.t += 1
                macro_interval += 1
            
            macro_post_transition_data = {
                "actions": macro_ids,
                "reward": [(macro_return,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)]
            }
            self.macro_batch.update(macro_post_transition_data, ts=self.macro_t)
            self.macro_t += 1


        all_avail_actions = self.env.get_avail_actions()
        last_macro_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.get_avail_macro_ids(all_avail_actions)],
            "obs": [self.env.get_obs()],
            "goal_obs": [self.env.get_goal_feats()],
            "adjacency_matrix":[self.env.get_adjacency_matrix()]
        }
        self.macro_batch.update(last_macro_data, ts=self.macro_t)
        
        # Select goals in the last stored state
        macro_ids = self.macro_mac.select_actions(self.macro_batch, t_ep=self.macro_t, t_env=self.t_macro, test_mode=test_mode)
        self.macro_batch.update({"actions": macro_ids}, ts=self.macro_t)


        agent_obs_feats = self.env.get_obs_feats()
        avail_move_actions_ls, avail_a_actions_ls = self.get_avail_move_a_actions(all_avail_actions)
        last_move_data = {
            "state": [self.env.get_state()],
            "avail_actions": [avail_move_actions_ls],
            "obs": [self.env.get_obs()],
            "adjacency_matrix":[self.env.get_adjacency_matrix()],
            "move_feats":[agent_obs_feats[0]],
            "enemy_feats":[agent_obs_feats[1]],
            "ally_feats":[agent_obs_feats[2]],
            "own_feats":[agent_obs_feats[3]]
        }
        self.move_batch.update(last_move_data, ts=self.t)

        # Select actions in the last stored state
        move_actions = self.move_mac.select_actions(self.move_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.move_batch.update({"actions": move_actions}, ts=self.t)

        last_a_data = {
            "state": [self.env.get_state()],
            "avail_actions": [avail_a_actions_ls],
            "obs": [self.env.get_obs()],
            "adjacency_matrix":[self.env.get_adjacency_matrix()],
            "move_feats":[agent_obs_feats[0]],
            "enemy_feats":[agent_obs_feats[1]],
            "ally_feats":[agent_obs_feats[2]],
            "own_feats":[agent_obs_feats[3]]
        }
        self.action_batch.update(last_a_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.action_mac.select_actions(self.action_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.action_batch.update({"actions": a_actions}, ts=self.t)


        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.t_macro += self.macro_t

        cur_returns.append(episode_return)

        # if test_mode:
            # self.logger.console_logger.info("test mode is True")
             # self.logger.console_logger.info("collect map data ...")
            # for idx, value in enumerate(self.map_data):
            #     np.save('{}_{}.npy'.format(self.args.env_args['map_name'], idx), np.array(value))
            #     self.logger.console_logger.info("save map data success!")

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.action_mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.action_mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.move_batch, self.action_batch, self.macro_batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
