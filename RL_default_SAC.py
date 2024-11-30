"""
author: huihang@mail.ustc.edu.cn
date: 2024-10-20
Setting:
    Env:        UVA/Padova simulator
    Reward:     magni_reward + reward bias(100), standardization
    Action:     action_cap(0.2) + action_bias + action_scale [0, 0.2], standardization
    State:      standardization
    Algorithm:  default SAC
    Tasks:      Single task training
Performance: 
    Well
"""

# %% Import libraries
import os, sys
from datetime import datetime
import numpy as np
import torch
import json
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt

work_dir = "diabetes-code/"
os.chdir(work_dir)  # set the working directory
sys.path.append(os.path.abspath(work_dir))
sys.path.append(os.path.abspath(work_dir + "utils/"))

from utils.BGEnv.UVAPEnv.envs.simglucose_gym_env import DeepSACT1DEnv
from utils.BGEnv.UVAPEnv.envs import reward_functions
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results
from stable_baselines3.common.monitor import Monitor


device = torch.device("cpu")

log_dir = "./results/"
patient_name = "adult#001"
time_start = datetime.now().strftime("%Y-%m-%d_%H-%M")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    os.makedirs(log_dir + "/tensorboard/")
if not os.path.exists(log_dir + time_start):
    os.makedirs(log_dir + time_start + "/model/")
    os.makedirs(log_dir + time_start + f"/callback_{patient_name}/")
    os.makedirs(log_dir + time_start + f"/eval/{patient_name}")

verbose = True


# %% Callback, wrapper
class SaveOnBestTrainingCallback(BaseCallback):
    """
    Stable Baselines3 loads episode statistics from a monitoring file, which is typically updated at the end of each `episode` when using the Monitor wrapper in your environment.
    """

    def __init__(self, check_freq, log_dir, patient_name, verbose=1):
        super(SaveOnBestTrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_episode_record = -np.inf
        self.patient_name = patient_name
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Load results using load_results which returns a DataFrame
            df = load_results(
                self.log_dir
            )  # each row is an episode include columns: 'r' cumulative reward, 'l' length, 't' time in seconds
            if not df.empty and len(df) >= 20:
                # max_length = df['l'].max()  # Find the maximum episode length
                avg_episode_record = (
                    df["r"].iloc[-20:].mean()
                )  # Find the best episode record by taking the average of the last 20 episodes
                if avg_episode_record > self.best_episode_record:
                    self.best_episode_record = avg_episode_record  # Update best length
                    # Save the new best model
                    self.model.save(
                        os.path.join(self.log_dir, "best_model_" + self.patient_name)
                    )
                    if self.verbose:
                        print(
                            f"New best episode record: {self.best_episode_record} - model saved at {self.log_dir}"
                        )
        return True


# Custom logging wrapper
class CustomWrapper(gym.Wrapper):
    def __init__(self, env, log_dir):
        super(CustomWrapper, self).__init__(env)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_actions = []
        self.episode_states = []
        self.current_rewards = []
        self.current_actions = []
        self.current_states = []
        self.episode_start_time = None
        self.episode_end_time = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.episode_start_time = datetime.now()
        self.current_rewards = []
        self.current_actions = []
        self.current_states = []
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_rewards.append(reward)
        self.current_actions.append(action)
        self.current_states.append(obs)
        if truncated:
            self.episode_end_time = datetime.now()
            self.episode_rewards.append(sum(self.current_rewards))
            self.episode_lengths.append(len(self.current_rewards))
            self.episode_actions.append(self.current_actions)
            self.episode_states.append(self.current_states)
            self._log_episode()
            self._plot_episode()
        obs = self._normalize_state(obs)
        reward = reward / 100
        return obs, reward, done, truncated, info

    def _log_episode(self):
        episode_data = {
            "rewards": self.current_rewards,
            "actions": self.current_actions,
            "states": self.current_states,
            "length": len(self.current_rewards),
            "start_time": self.episode_start_time,
            "end_time": self.episode_end_time,
        }
        log_file = os.path.join(
            self.log_dir, f"episode_{len(self.episode_rewards)}.json"
        )
        with open(log_file, "a") as f:
            json.dump(episode_data, f, default=str)

    def _plot_episode(self):
        df = pd.DataFrame()
        df["Time"] = pd.Series(self.env.env.env.time_hist)
        df["BG"] = pd.Series(self.env.env.env.BG_hist)
        df["CGM"] = pd.Series(self.env.env.env.CGM_hist)
        df["CHO"] = pd.Series(self.env.env.env.CHO_hist)
        df["insulin"] = pd.Series(self.env.env.env.insulin_hist)
        df["LBGI"] = pd.Series(self.env.env.env.LBGI_hist)
        df["HBGI"] = pd.Series(self.env.env.env.HBGI_hist)
        df["Risk"] = pd.Series(self.env.env.env.risk_hist)
        df["Magni_Risk"] = pd.Series(self.env.env.env.magni_risk_hist)

        df["Time"] = pd.to_datetime(df["Time"])
        # turn time into minutes
        df["Time"] = (
            (df["Time"] - df["Time"].iloc[0]).dt.total_seconds() / 60 / 60
        )  # hours

        glucose_levels = df["CGM"]
        insulin_rates = df["insulin"]  # Assuming this column exists
        rewards_steps = df["Magni_Risk"]  # Assuming this represents some form of reward

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
        ax2.set_xticks([8, 12, 18, 24, 32, 36, 42, 48, 56, 60, 66, 72, 80, 84, 90, 96])
        ax2.set_ylim(0, 0.5)  # insulin rate range

        # Plot for rewards
        plt.subplot(2, 1, 2)
        plt.plot(df["Time"], rewards_steps, color="blue", label="Risk")
        plt.xlabel("Time (Days)")
        plt.ylabel("Risk")
        # show cumulative reward as a note on the plot
        plt.text(
            0.5,
            0.5,
            f"Total Risk: {np.sum(rewards_steps):.4f}",
            fontsize=12,
            transform=plt.gcf().transFigure,
        )
        # show patient name as a note on the plot
        plt.text(
            0.2,
            0.5,
            f"Patient: {self.env.env.patient_name}",
            fontsize=12,
            transform=plt.gcf().transFigure,
        )
        plt.xlim(x_range)
        plt.xticks([8, 12, 18, 24, 32, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96])

        plt.tight_layout()

        plt.suptitle("Glucose Level and Insulin Infusion Rate Over Time")
        cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"{self.log_dir}{cur_time}.png")
        plt.close()

    def _normalize_state(self, state):
        # the state contains 48*2 elements, the first 48 elements are recent glucose levels and the last 48 elements are recent insulin rates
        # we normalize the state by dividing the glucose levels by 1000 and the insulin rates by 0.4, which are the maximum values of the two types of data
        if len(state) == 96:
            state[:48] = state[:48] / 500
            state[48:] = state[48:] / 0.4
        else:
            raise ValueError("The state should contain 96 elements")
        return state


def simulate_once(patient_name):
    env = DeepSACT1DEnv(
        reward_fun=reward_functions.magni_reward,
        bw_meals=True,
        n_hours=4,
        time_limit=24 * 60 / 5 * 1,  # 24h with 5 minute per step
        hist_init=True,
        update_seed_on_reset=True,  # Realistic Variation in training data
        action_cap=0.5,  # remove the action cap
        action_bias=1,  # action = (action + action_bias) * action_scale
        action_scale=2 / 20,
        reward_bias=100,  # reward = reward + reward_bias, to encourage a long episode
        termination_penalty=1e4,  # default: no termination penalty
        patient_name=patient_name,
        universal=False,  # use random patient
        source_dir=work_dir + "utils/",
        log_dir=log_dir + time_start + "/train/",
        verbose=verbose,
    )
    env = Monitor(
        env,
        log_dir + time_start + "/callback_" + patient_name + "/",
        allow_early_resets=True,
    )
    env = CustomWrapper(env, log_dir + time_start + "/callback_" + patient_name + "/")
    callback = SaveOnBestTrainingCallback(
        check_freq=200,
        log_dir=log_dir + time_start + "/callback_" + patient_name + "/",
        verbose=True,
        patient_name=patient_name,
    )

    # Train
    if verbose:
        print("RL_v2.py - Train")

    model_flag = "SAC"  # "PPO", "SAC"

    if model_flag == "PPO":
        # PPO (Proximal Policy Optimization) algorithm
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=512,
            gamma=0.99,
            learning_rate=3e-4,
            device=device,
            tensorboard_log=log_dir + "tensorboard/",
        )
    elif model_flag == "SAC":
        # Soft Actor-Critic (SAC) algorithm
        model = SAC(
            "MlpPolicy",
            env,
            buffer_size=int(1e6),
            batch_size=512,
            gamma=0.99,
            verbose=0,  # 0: no output, 1: progress bar, 2: one line per step
            learning_rate=0.0003,
            tensorboard_log=log_dir + "tensorboard/",
            device=device,
        )

    # Train the agent
    total_timesteps = 1 * 100000  # 50 * 10000
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # Save the trained agent
    if True:
        file_name = (
            log_dir + time_start + "/model" + "/trained_" + patient_name + ".zip"
        )
        model.save(file_name)
        if verbose:
            print(f"Trained model saved at {file_name}")
        if not os.path.exists(
            log_dir
            + time_start
            + "/callback_"
            + patient_name
            + "/"
            + "best_model_"
            + patient_name
            + ".zip"
        ):
            model.save(
                log_dir
                + time_start
                + "/callback_"
                + patient_name
                + "/"
                + "best_model_"
                + patient_name
            )

    # Evaluation
    if verbose:
        print("RL_v2.py - Evaluation")

    def evaluate(loaded_model, type="trained"):
        time_limit = 24 * 60 / 5 * 3  # 3 days
        deterministic = False if type == "online" else True
        env = DeepSACT1DEnv(
            reward_fun=reward_functions.magni_reward,
            bw_meals=True,
            n_hours=4,
            time_limit=time_limit,  # 24h with 5 minute per step
            update_seed_on_reset=True,  # Realistic Variation in training data
            hist_init=True,
            action_cap=0.5,  # remove the action cap
            action_bias=1,
            action_scale=2 / 20,
            reward_bias=100,
            patient_name=patient_name,
            universal=False,  # use random patient
            source_dir=work_dir + "utils/",
            log_dir=log_dir + time_start + "/eval/" + type,
            verbose=verbose,
        )
        env = Monitor(
            env,
            log_dir + time_start + "/eval/" + patient_name + "/",
            allow_early_resets=True,
        )
        env = CustomWrapper(env, log_dir + time_start + "/eval/" + patient_name)
        _ = env.reset()

        total_time = int(time_limit)  # one day 288 sample points
        # Initialize lists to store glucose levels, insulin rates, and time steps
        flag_failure = False
        cur_state = env.unwrapped.get_state()
        for t in range(total_time):  # one day 288 sample points
            action, _states = loaded_model.predict(
                cur_state.astype(np.float32), deterministic=deterministic
            )
            cur_state, rewards, done, truncated, info = env.step(action)
            flag_failure = flag_failure or info["flag_failure"]

            if False:
                env.render()

            if truncated:
                env._plot_episode()
                break
                # obs = env.reset()  # Reset the environment if done

        if verbose:
            print(f"End of evaluation: {type}, flag_failure: {flag_failure}")

        # obtain the simulation results
        df = pd.DataFrame()
        df["Time"] = pd.Series(env.unwrapped.env.time_hist)
        df["BG"] = pd.Series(env.unwrapped.env.BG_hist)
        df["CGM"] = pd.Series(env.unwrapped.env.CGM_hist)
        df["CHO"] = pd.Series(env.unwrapped.env.CHO_hist)
        df["insulin"] = pd.Series(env.unwrapped.env.insulin_hist)
        df["LBGI"] = pd.Series(env.unwrapped.env.LBGI_hist)
        df["HBGI"] = pd.Series(env.unwrapped.env.HBGI_hist)
        df["Risk"] = pd.Series(env.unwrapped.env.risk_hist)
        df["Magni_Risk"] = pd.Series(env.unwrapped.env.magni_risk_hist)
        df["Time"] = pd.to_datetime(df["Time"])
        # transfer time into minutes or hours or days
        df["Time"] = (
            (df["Time"] - df["Time"].iloc[0]).dt.total_seconds() / 60 / 60
        )  # hours

        #  Show regions
        # calculate the number of time points where the glucose level is below 70 mg/dL, and between 70 and 180 mg/dL and above 180 mg/dL
        rewards_steps = df["Magni_Risk"]  # Assuming this represents some form of reward
        low = df["BG"][df["BG"] < 70].count()
        normal = df["BG"][(df["BG"] >= 70) & (df["CGM"] <= 180)].count()
        high = df["BG"][df["BG"] > 180].count()

        # get their ratio (percentage on the total time points)
        low_ratio = low / len(df["BG"]) * 100
        normal_ratio = normal / len(df["BG"]) * 100
        high_ratio = high / len(df["BG"]) * 100

        if False:
            print(
                f"Risk: {np.sum(rewards_steps):.4f}, Low: {low_ratio:.2f}%, Normal: {normal_ratio:.2f}%, High: {high_ratio:.2f}%, Failure: {flag_failure}"
            )
        else:
            # save the results into the output.csv file
            output_file = log_dir + time_start + "/eval/" + "output_" + type + ".csv"
            res = [
                patient_name,
                np.sum(rewards_steps),
                low_ratio,
                normal_ratio,
                high_ratio,
                int(flag_failure),
            ]
            with open(output_file, "a") as f:
                f.write(",".join([str(r) for r in res]))
                f.write("\n")

    evaluate(model, "online")  # evaluate the online model
    evaluate(model, "trained")  # evaluate the trained model

    # if model is SAC, we load via SAC.load, else we load via PPO.load
    if isinstance(model, SAC):
        saved_best_model = SAC.load(
            log_dir
            + time_start
            + "/callback_"
            + patient_name
            + "/"
            + "best_model_"
            + patient_name,
            # env=env,
        )
    elif isinstance(model, PPO):
        saved_best_model = PPO.load(
            log_dir
            + time_start
            + "/callback_"
            + patient_name
            + "/best_model_"
            + patient_name,
            # env=env,
        )
    evaluate(saved_best_model, "best")


if __name__ == "__main__":
    simulate_once(patient_name)
    print(
        "Time escaped: ",
        datetime.now() - datetime.strptime(time_start, "%Y-%m-%d_%H-%M"),
    )
