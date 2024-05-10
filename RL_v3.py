"""
This is going to use custom SAC policy and feature extractor.

Modifications:
  sac/sac.py
  sac/policies.py
author: huihang@mail.ustc.edu.cn
date: 2024-03-23
"""
# %% Import libraries
import os
import sys
import platform
system_type = platform.system().lower()
if system_type == 'linux':
  work_dir = '/data/huihang/Codes/Python/rl-control/'
else:
  work_dir = '/Users/huihang/Library/CloudStorage/OneDrive-Personal/Repository/RL-Control/'
os.chdir(work_dir) # set the working directory
sys.path.append(os.path.abspath(work_dir))
sys.path.append(os.path.abspath(work_dir+'utils/'))

from datetime import datetime
time_start = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
import numpy as np
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True) # for debug
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt

from utils.BGEnv.UVAPEnv.envs.simglucose_gym_env import DeepSACT1DEnv
from utils.BGEnv.UVAPEnv.envs import reward_functions

from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.policies import register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # for multiple environments

import cProfile
import pstats
import pandas as pd

class SaveOnBestTrainingCallback(BaseCallback):
  """
  Stable Baselines3 loads episode statistics from a monitoring file, which is typically updated at the end of each `episode` when using the Monitor wrapper in your environment. 
  """
  def __init__(self, check_freq, eval_env, log_dir, verbose=1):
    super(SaveOnBestTrainingCallback, self).__init__(verbose)
    self.eval_env = eval_env
    self.check_freq = check_freq # trajectory/timestep number to check for best model
    self.log_dir = log_dir
    self.best_reward = -np.inf
    self.episode_count = 0  # Track the number of episodes processed

  def _init_callback(self):
    if self.eval_env is None:
      raise ValueError("An evaluation environment must be provided")

  def _on_training_start(self):
    self.evaluate_model()  # Evaluate the model at the start of training

  def _on_step(self) -> bool:
    # Check if the 'done' flag was True in the info dictionary which indicates the end of an episode
    truncated_flag = [self.locals['infos'][t]['TimeLimit.truncated'] for t in range(len(self.locals['infos']))]
    if any(self.locals['dones']) or any(truncated_flag):
        self.episode_count += 1
        # Check if the number of episodes is enough to trigger evaluation
        if self.episode_count % self.check_freq == 0:
          self.evaluate_model()
    return True

  def evaluate_model(self):

    total_reward = 0.0
    done = truncated = False
    _ = self.eval_env.reset()
    obs = self.eval_env.get_state()
    while (not truncated) and (not done): 
      action, _ = self.model.predict(obs.astype(np.float32), deterministic=True)
      obs, reward, done, truncated, info = self.eval_env.step(action)
      total_reward += reward
    
    print(f"\tEval the model at timestep count: {self.n_calls}, episode count: {self.episode_count}, total_reward: {total_reward}")

    # print the results
    df = pd.DataFrame()
    df["Time"] = pd.Series(self.eval_env.env.time_hist)
    df["BG"] = pd.Series(self.eval_env.env.BG_hist)
    df["CGM"] = pd.Series(self.eval_env.env.CGM_hist)
    df["CHO"] = pd.Series(self.eval_env.env.CHO_hist)
    df["insulin"] = pd.Series(self.eval_env.env.insulin_hist)
    df["LBGI"] = pd.Series(self.eval_env.env.LBGI_hist)
    df["HBGI"] = pd.Series(self.eval_env.env.HBGI_hist)
    df["Risk"] = pd.Series(self.eval_env.env.risk_hist)
    df["Magni_Risk"] = pd.Series(self.eval_env.env.magni_risk_hist)

    df["Time"] = pd.to_datetime(df["Time"])
    # turn time into minutes
    df["Time"] = (
        (df["Time"] - df["Time"].iloc[0]).dt.total_seconds() / 60 / 60
    )  # hours

    glucose_levels = df["CGM"]
    insulin_rates = df["insulin"]  # Assuming this column exists
    rewards_steps = df[
        "Magni_Risk"
    ]  # Assuming this represents some form of reward

    meal_times = df["Time"][df["CHO"] > 0]
    meal_amounts = df["CHO"][df["CHO"] > 0]
    x_range = (0, 91)  # Set the x-axis range

    plt.figure(figsize=(12, 12))
    ax1 = plt.subplot(2, 1, 1)

    color = "tab:red"
    ax1.set_xlabel("Time (Days)")
    ax1.set_ylabel("Glucose Level (mg/dL)", color=color)
    ax1.plot(df["Time"], df["CGM"], color=color, label="Glucose Level")
    ax1.plot(df["Time"], df["BG"], color="black", linestyle="--", label="BG")
    for meal_time in meal_times:
        ax1.axvline(x=meal_time, color="blue", linestyle="--", linewidth=1)
    ax1.tick_params(axis="y", labelcolor=color)

    ax1.axhline(y=80, color="green", linestyle="--", label="Lower Target")
    ax1.axhline(y=140, color="green", linestyle="--")
    ax1.set_xlim(x_range)
    ax1.set_ylim(10, 300)

    ax2 = ax1.twinx()  # Second axes for insulin rates
    color = "tab:blue"
    ax2.set_ylabel("Insulin Infusion Rate", color=color)
    ax2.plot(df["Time"], insulin_rates, color=color, alpha=0.2)
    ax2.tick_params(axis="y", labelcolor=color)
    # please set the x-axis to display the following values: 8, 12, 18, 24, 32, 36, 42, 48, 56, 60, 66, 72
    ax2.set_xticks(
        [8, 12, 18, 24, 32, 36, 42, 48, 56, 60, 66, 72, 80, 84, 90, 96]
    )
    ax2.set_ylim(0, 1.0)

    # Plot for rewards
    plt.subplot(2, 1, 2)
    plt.plot(df["Time"], rewards_steps, color="blue", label="Risk")
    plt.xlabel("Time (Days)")
    plt.ylabel("Risk")
    plt.xlim(x_range)
    # please set_xticks([8, 12, 18, 24, 32, 36, 42, 48, 56, 60, 66, 72, 80, 84, 90, 96])
    plt.xticks([8, 12, 18, 24, 32, 36, 42, 48, 56, 60, 66, 72, 80, 84, 90, 96])

    plt.tight_layout()

    # Adjust title and save logic according to your needs
    plt.suptitle("Glucose Level and Insulin Infusion Rate Over Time")
    time_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(
        f"./backup/fig/debug/eval/RL_v3_debug_glucose_insulin.{time_start}.png"
    )
    # plt.show()
    plt.close()
    
    if total_reward > self.best_reward:
      self.best_reward = total_reward
      self.model.save(os.path.join(self.log_dir, 'best_model'))
      if self.verbose > 0:
        print(f"New best model saved with reward: {self.best_reward}")
    # print(f"End Step: {self.n_calls}, evaluate the model")
    # raise NotImplementedError("Debug at the end of evaluation")

def make_env():
  def _init():
    env = DeepSACT1DEnv(reward_fun=reward_functions.magni_reward, 
                        bw_meals=True, 
                        n_hours=4,
                        time_limit=24*60/5*1,
                        hist_init=True,
                        action_cap=None,
                        patient_name='adolescent#002',
                        universal=False,
                        source_dir=work_dir+'utils/')
    env = Monitor(env, log_dir+'monitor')
    return env
  return _init

np.random.seed(2024)
torch.manual_seed(2025)

log_dir = "./backup/"

if __name__ == '__main__':
  if False:
    # Use only one environment
    env = DeepSACT1DEnv(
      reward_fun=reward_functions.magni_reward, 
      bw_meals=True, 
      n_hours=4,
      time_limit=24*60/5*1, # 24h with 5 minute per step
      hist_init=True,
      action_cap=None, # remove the action cap
      patient_name='adolescent#001',
      universal=False, # use random patient
      source_dir=work_dir+'utils/'
    )
    env = Monitor(env, log_dir+'monitor')
  else: 
    # Create multiple environments
    env = DummyVecEnv([make_env() for _ in range(4)])
    # env = SubprocVecEnv([make_env() for _ in range(4)])

  eval_env = DeepSACT1DEnv(
    reward_fun=reward_functions.magni_reward, 
    bw_meals=True, 
    n_hours=4,
    time_limit=24*60/5*2, # 48h with 5 minute per step, different from training env
    hist_init=True,
    action_cap=None, # remove the action cap
    patient_name='adolescent#002',
    universal=False, # use random patient
    source_dir=work_dir+'utils/'
  )
  callback = SaveOnBestTrainingCallback(check_freq=10, eval_env=eval_env, log_dir=log_dir+'monitor')


  # %% Train

  # Soft Actor-Critic (SAC) algorithm
  model = SAC("MlpPolicy", env, 
              buffer_size=int(1e6), # default: 1e6 
              batch_size=128, # default: 256
              gamma=0.99,
              verbose=0, 
              learning_rate= 0.0003, # default: 0.0003
              tensorboard_log=log_dir+'tensorboard/', 
              device=device)
  # model = SAC.load('./backup/model/diabetes.2024-04-18_16-38-53.agent', env=env) # for continue training

  # CustomSACPolicy, requires a custom policy and a custom feature extractor
  # model = SAC(CustomActorCriticPolicy, env, 
  #             buffer_size=int(1e6),
  #             batch_size=256,
  #             gamma=0.99,
  #             verbose=1, 
  #             learning_rate=0.0003, 
  #             tensorboard_log=log_dir+'tensorboard/', 
  #             device=device)

  # # Create a profiler instance
  # profiler = cProfile.Profile()
  # profiler.enable()

  # Train the agent
  total_timesteps = int(10 * 10000)  # 300 epochs * 1000 timesteps per epoch
  model.learn(total_timesteps=total_timesteps, 
              callback=callback, 
              progress_bar=True)

  # # Disable the profiler and print out stats
  # profiler.disable()
  # stats = pstats.Stats(profiler).sort_stats('cumtime')
  # # Define the file path for the output
  # output_file_path = './profile_results.txt'
  # with open(output_file_path, 'w') as file:
  #     stats.stream = file
  #     stats.print_stats()

  # Save the agent
  if True:
    file_name = './backup/model/diabetes.' + time_start + '.agent'
    print(f"Model saved at {file_name}")
    model.save(file_name)

  # %% Evaluation

  # Load the agent
  agent = 'best' # 'trained' or 'best' or 'diabetes.2024-04-18_16-38-53'
  if agent == 'trained':
    loaded_model = model
  elif agent == 'best':
    model_path = './backup/monitor/best_model.zip'
    loaded_model = SAC.load(model_path)
  else:
    loaded_model = SAC.load('./backup/model/' + agent + '.agent', env=env)

  # np.random.seed(2024)
  time_limit = 24*60/5*3
  deterministic = True
  env = DeepSACT1DEnv(
    reward_fun=reward_functions.magni_reward, 
    bw_meals=True, 
    n_hours=4,
    time_limit=time_limit, # 24h with 5 minute per step
    hist_init=True,
    action_cap=None, # remove the action cap
    patient_name='adolescent#001',
    universal=False, # use random patient
    source_dir=work_dir+'utils/'
  )
  _ = env.reset()

  total_time = int(time_limit) # one day 288 sample points
  # Initialize lists to store glucose levels, insulin rates, and time steps
  glucose_levels = []
  insulin_rates = []
  time_steps = []
  rewards_steps = []

  cur_state = env.unwrapped.get_state()

  for t in range(total_time): # one day 288 sample points
    action, _states = loaded_model.predict(cur_state.astype(np.float32), deterministic=deterministic)
    cur_state, rewards, done, truncated, info = env.step(action)

    if False: env.render()

    # Append current glucose level, insulin rate, and time step
    glucose_levels.append(cur_state[47])
    insulin_rates.append(action[0])
    time_steps.append(t/60*5)
    rewards_steps.append(rewards)

    if done or truncated:
      break
      obs = env.reset()  # Reset the environment if done

  print(f"Cumulated rewards: {np.sum(rewards_steps)}")

  # %% Plot
  import pandas as pd
  from datetime import datetime

  df = pd.DataFrame()
  df['Time'] = pd.Series(env.unwrapped.env.time_hist)
  df['BG'] = pd.Series(env.unwrapped.env.BG_hist)
  df['CGM'] = pd.Series(env.unwrapped.env.CGM_hist)
  df['CHO'] = pd.Series(env.unwrapped.env.CHO_hist)
  df['insulin'] = pd.Series(env.unwrapped.env.insulin_hist)
  df['LBGI'] = pd.Series(env.unwrapped.env.LBGI_hist)
  df['HBGI'] = pd.Series(env.unwrapped.env.HBGI_hist)
  df['Risk'] = pd.Series(env.unwrapped.env.risk_hist)
  df['Magni_Risk'] = pd.Series(env.unwrapped.env.magni_risk_hist)

  df['Time'] = pd.to_datetime(df['Time'])
  # turn time into minutes
  df['Time'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds() / 60 / 60 # hours

  glucose_levels = df['CGM']
  insulin_rates = df['insulin']
  rewards_steps = df['Magni_Risk']

  meal_times = df['Time'][df['CHO'] > 0]
  meal_amounts = df['CHO'][df['CHO'] > 0]
  x_range = (0, 91)  # Set the x-axis range

  plt.figure(figsize=(12, 12))
  ax1 = plt.subplot(2, 1, 1)

  color = 'tab:red'
  ax1.set_xlabel('Time (Days)')
  ax1.set_ylabel('Glucose Level (mg/dL)', color=color)
  ax1.plot(df['Time'], df['CGM'], color=color, label='Glucose Level')
  ax1.plot(df['Time'], df['BG'], color='black', linestyle='--', label='BG')
  for meal_time in meal_times:
    ax1.axvline(x=meal_time, color='blue', linestyle='--', linewidth=1)
  ax1.tick_params(axis='y', labelcolor=color)

  ax1.axhline(y=80, color='green', linestyle='--', label='Lower Target')
  ax1.axhline(y=140, color='green', linestyle='--')
  ax1.set_xlim(x_range)
  ax1.set_ylim(10, 300)

  ax2 = ax1.twinx()  # Second axes for insulin rates
  color = 'tab:blue'
  ax2.set_ylabel('Insulin Infusion Rate', color=color)
  ax2.plot(df['Time'], insulin_rates, color=color, alpha=0.2)
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.set_xticks([8, 12, 18, 24, 32, 36, 42, 48, 56, 60, 66, 72, 80, 84, 90, 96])
  ax2.set_ylim(0, 1.0)

  # Plot for rewards
  plt.subplot(2, 1, 2)
  plt.plot(df['Time'], rewards_steps, color='blue', label='Risk')
  plt.xlabel('Time (Days)')
  plt.ylabel('Risk')
  plt.xlim(x_range)
  # please set_xticks([8, 12, 18, 24, 32, 36, 42, 48, 56, 60, 66, 72, 80, 84, 90, 96])
  plt.xticks([8, 12, 18, 24, 32, 36, 42, 48, 56, 60, 66, 72, 80, 84, 90, 96])

  plt.tight_layout()

  # Adjust title and save logic according to your needs
  plt.suptitle('Glucose Level and Insulin Infusion Rate Over Time')
  time_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  plt.savefig(f'./backup/fig/RL_v3_glucose_insulin.{env.unwrapped.patient_name}.{time_start}.png')
  plt.show()

  # %% Show regions
  # calculate the number of time points where the glucose level is below 70 mg/dL, and between 70 and 180 mg/dL and above 180 mg/dL
  df = df[df['Time'] >= 24]

  low = df['BG'][df['BG'] < 70].count()
  normal = df['BG'][(df['BG'] >= 70) & (df['CGM'] <= 180)].count()
  high = df['BG'][df['BG'] > 180].count()

  # get their ratio (percentage on the total time points)
  low_ratio = low / len(df['BG']) * 100
  normal_ratio = normal / len(df['BG']) * 100
  high_ratio = high / len(df['BG']) * 100

  print(f"Length: {len(df['BG']) / 12:.2f} (hours), Low: {low_ratio:.2f}%, Normal: {normal_ratio:.2f}%, High: {high_ratio:.2f}%")

  # %%
