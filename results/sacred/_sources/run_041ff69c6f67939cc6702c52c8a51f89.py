import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    # print(env_info)
    args.a_action_size = env_info["n_actions"] - args.a_move_size + 1

    # Default/Base scheme   move scheme
    move_scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "pure_obs":{"vshape": env_info["obs_shape"], "group": "agents"},                            # raw obs size
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},       # raw obs + goal size
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (args.a_move_size,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},                                             # reward not shared micro policy
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "adjacency_matrix": {"vshape":(env_info["n_agents"], env_info["n_agents"])},
        "move_feats":{"vshape":(env_info["n_agents"], env_info["move_feats_size"])},
        "enemy_feats":{"vshape":(env_info["n_agents"], env_info["enemy_feats_size"])},
        "ally_feats":{"vshape":(env_info["n_agents"], env_info["ally_feats_size"])},
        "own_feats":{"vshape":(env_info["n_agents"], env_info["own_feats_size"])},
        "move_feats_size":{"vshape":env_info["move_feats_size"]},
        "enemy_feats_size":{"vshape":env_info["enemy_feats_size"]},
        "ally_feats_size":{"vshape":env_info["ally_feats_size"]},
        "own_feats_size":{"vshape":env_info["own_feats_size"]},
        "goal_obs":{"vshape":env_info["goal_shape"], "group": "agents"}
    }

    # Default/Base scheme action scheme
    action_scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "pure_obs":{"vshape": env_info["obs_shape"], "group": "agents"},                            # raw obs size
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},       # raw obs + goal size
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (args.a_action_size,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},                                             # reward not shared micro policy
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "adjacency_matrix": {"vshape":(env_info["n_agents"], env_info["n_agents"])},
        "move_feats":{"vshape":(env_info["n_agents"], env_info["move_feats_size"])},
        "enemy_feats":{"vshape":(env_info["n_agents"], env_info["enemy_feats_size"])},
        "ally_feats":{"vshape":(env_info["n_agents"], env_info["ally_feats_size"])},
        "own_feats":{"vshape":(env_info["n_agents"], env_info["own_feats_size"])},
        "move_feats_size":{"vshape":env_info["move_feats_size"]},
        "enemy_feats_size":{"vshape":env_info["enemy_feats_size"]},
        "ally_feats_size":{"vshape":env_info["ally_feats_size"]},
        "own_feats_size":{"vshape":env_info["own_feats_size"]},
        "goal_obs":{"vshape":env_info["goal_shape"], "group": "agents"}
    }

    # goal scheme
    macro_scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (args.a_macro_size,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "adjacency_matrix": {"vshape":(env_info["n_agents"], env_info["n_agents"])},
        "move_feats":{"vshape":(env_info["n_agents"], env_info["move_feats_size"])},
        "enemy_feats":{"vshape":(env_info["n_agents"], env_info["enemy_feats_size"])},
        "ally_feats":{"vshape":(env_info["n_agents"], env_info["ally_feats_size"])},
        "own_feats":{"vshape":(env_info["n_agents"], env_info["own_feats_size"])},
        "move_feats_size":{"vshape":env_info["move_feats_size"]},
        "enemy_feats_size":{"vshape":env_info["enemy_feats_size"]},
        "ally_feats_size":{"vshape":env_info["ally_feats_size"]},
        "own_feats_size":{"vshape":env_info["own_feats_size"]},
        "goal_obs":{"vshape":env_info["goal_shape"], "group": "agents"}

    }

    groups = {
        "agents": args.n_agents
    }

    move_preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.a_move_size)])
    }

    action_preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.a_action_size)])
    }

    macro_preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.a_macro_size)])          # macro action size : goal num
    }


    move_buffer = ReplayBuffer(move_scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=move_preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    action_buffer = ReplayBuffer(action_scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=action_preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)


    macro_buffer = ReplayBuffer(macro_scheme, groups, args.buffer_size, int(env_info["episode_limit"] / args.min_horizon) + 1,
                          preprocess=macro_preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here  move controller
    move_mac = mac_REGISTRY[args.mac]("micro policy move", move_buffer.scheme, groups, args)

    # Setup multiagent controller here action controller
    action_mac = mac_REGISTRY[args.mac]("micro policy action", action_buffer.scheme, groups, args)

    # macro controller here 2021/06/13
    macro_mac = mac_REGISTRY[args.mac]("macro policy", macro_buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(macro_scheme=macro_scheme, move_scheme=move_scheme, action_scheme=action_scheme, groups=groups, move_preprocess=move_preprocess, action_preprocess=action_preprocess, macro_preprocess=macro_preprocess, move_mac=move_mac, action_mac=action_mac, macro_mac=macro_mac)

    # Learner
    move_learner = le_REGISTRY[args.learner]("micro policy move", move_mac, move_buffer.scheme, logger, args)
    action_learner = le_REGISTRY[args.learner]("micro policy action", action_mac, action_buffer.scheme, logger, args)
    macro_learner = le_REGISTRY[args.learner]("macro policy", macro_mac, macro_buffer.scheme, logger, args)

    if args.use_cuda:
        move_learner.cuda()
        action_learner.cuda()
        macro_learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
        macro_model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))                  # 待修改

        logger.console_logger.info("Loading model from {}".format(model_path))
        move_learner.load_models(model_path)
        action_learner.load_models(model_path)
        macro_learner.load_models(macro_model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # logger.console_logger.info("runner in goal exploration phase!")
    # runner.goal_exploration(logger, test_mode=False)
    # logger.console_logger.info("runner end goal exploration phase!")

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        move_batch, action_batch, macro_batch = runner.run(test_mode=False)
        move_buffer.insert_episode_batch(move_batch)
        action_buffer.insert_episode_batch(action_batch)
        macro_buffer.insert_episode_batch(macro_batch)
        
        # # train micro policy move
        if move_buffer.can_sample(args.batch_size):
            move_episode_sample = move_buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = move_episode_sample.max_t_filled()
            move_episode_sample = move_episode_sample[:, :max_ep_t]

            if move_episode_sample.device != args.device:
                move_episode_sample.to(args.device)

            move_learner.train(move_episode_sample, runner.t_env, episode)

        # train micro policy action
        if action_buffer.can_sample(args.batch_size):
            action_episode_sample = action_buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = action_episode_sample.max_t_filled()
            action_episode_sample = action_episode_sample[:, :max_ep_t]

            if action_episode_sample.device != args.device:
                action_episode_sample.to(args.device)

            action_learner.train(action_episode_sample, runner.t_env, episode)


        # train macro policy
        if macro_buffer.can_sample(args.batch_size):
            macro_episode_sample = macro_buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = macro_episode_sample.max_t_filled()
            macro_episode_sample = macro_episode_sample[:, :max_ep_t]

            if macro_episode_sample.device != args.device:
                macro_episode_sample.to(args.device)

            macro_learner.train(macro_episode_sample, runner.t_macro, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            move_learner.save_models(save_path)
            action_learner.save_models(save_path)
            macro_learner.save_models(save_path)


        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
