from BGEnv.UVAPEnv.simulation.env import T1DSimEnv
from BGEnv.UVAPEnv.patient.t1dpatient import T1DPatientNew
from BGEnv.UVAPEnv.sensor.cgm import CGMSensor
from BGEnv.UVAPEnv.actuator.pump import InsulinPump
from BGEnv.UVAPEnv.simulation.scenario_gen import (
    RandomBalancedScenario,
    SemiRandomBalancedScenario,
    CustomBalancedScenario,
)
from BGEnv.UVAPEnv.controller.base import Action
from BGEnv.UVAPEnv.analysis.risk import magni_risk_index
from BGEnv.UVAPEnv.envs.helpers import Seed
from BGEnv.UVAPEnv.envs import pid

import pandas as pd
import numpy as np
import joblib
import copy
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from datetime import datetime
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

"""
Huihang debug: 
    1. Modify simglucose_gym.py my_seed() at line 745. 
"""

# %%
class DeepSACT1DEnv(gym.Env):
    """
    A gym environment supporting SAC learning.
    Uses PID control for initialization.
    """

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        reward_fun,
        patient_name=None,
        seeds=None,
        reset_lim=None,
        time=False,  # include time informaction
        meal=False,
        gt=False,  # use_ground_truth, Oracle
        load=False,
        bw_meals=False,
        n_hours=4,  # 状态历史长度, 4 小时
        time_std=None,
        norm=False,  # 对 state 进行 normalize
        use_old_patient_env=False,
        action_cap=0.1,
        action_bias=0,
        action_scale=1,
        basal_scaling=216.0,  # basal_scaling is 43.2
        meal_announce=None,
        residual_basal=False,
        residual_bolus=False,
        residual_PID=False,
        fake_gt=False,
        fake_real=False,
        suppress_carbs=False,
        limited_gt=False,
        termination_penalty=1e5,
        weekly=False,
        use_model=False,
        model=None,
        model_device="cpu",
        update_seed_on_reset=False,
        deterministic_meal_size=False,
        deterministic_meal_time=False,
        deterministic_meal_occurrence=False,
        use_pid_load=False,
        hist_init=False,
        start_date=None,
        use_custom_meal=False,
        custom_meal_num=3,
        custom_meal_size=1,
        starting_glucose=None,
        harrison_benedict=False,
        restricted_carb=False,
        meal_duration=1,
        rolling_insulin_lim=None,
        universal=False,
        unrealistic=False,
        reward_bias=0,
        carb_error_std=0,
        carb_miss_prob=0,
        source_dir=None,
        time_limit=24 * 60 / 5 * 3, # huihang add this variable, in the fox paper it should be 20 days
        **kwargs,
    ):
        """
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        """
        super(DeepSACT1DEnv, self).__init__()
        self.source_dir = source_dir
        self.patient_para_file = "{}/BGEnv/UVAPEnv/params/vpatient_params.csv".format(
            self.source_dir
        )
        self.control_quest = "{}/BGEnv/UVAPEnv/params/Quest2.csv".format(
            self.source_dir
        )
        self.pid_para_file = "{}/BGEnv/UVAPEnv/params/pid_params.csv".format(
            self.source_dir
        )
        self.pid_env_path = "{}/BGEnv/UVAPEnv/params".format(self.source_dir)
        self.sensor_para_file = "{}/BGEnv/UVAPEnv/params/sensor_params.csv".format(
            self.source_dir
        )
        self.insulin_pump_para_file = "{}/BGEnv/UVAPEnv/params/pump_params.csv".format(
            self.source_dir
        )
        # reserving half of pop for testing
        self.universe = (
            ["child#0{}".format(str(i).zfill(2)) for i in range(1, 6)]
            + ["adolescent#0{}".format(str(i).zfill(2)) for i in range(1, 6)]
            + ["adult#0{}".format(str(i).zfill(2)) for i in range(1, 6)]
        )
        self.universal = universal
        if seeds is None:
            seed_list = self.my_seed()
            seeds = Seed(
                numpy_seed=seed_list[0],
                sensor_seed=seed_list[1],
                scenario_seed=seed_list[2],
            )
        if patient_name is None:
            if self.universal:
                patient_name = np.random.choice(self.universe)
            else:
                # patient_name = 'adolescent#001'
                patient_name = self.universe[11]
        # raise an error, show the seeds.numpy_seed
        # raise ValueError(seeds.numpy_seed)
        np.random.seed(seeds.numpy_seed)
        self.seeds = seeds
        self.sample_time = 5  # 5 minutes per sample
        self.day = int(
            1440 / self.sample_time
        )  # 1440 minutes per day, 288 samples per day
        self.state_hist = int(
            (n_hours * 60) / self.sample_time
        )  # state_hist 长度为, 即保留的历史数据长度, 是否应该是 4 小时 ??
        self.time = time
        self.meal = meal
        self.norm = norm
        self.gt = gt
        self.reward_fun = reward_fun
        self.reward_bias = reward_bias
        self.action_cap = action_cap
        self.action_bias = action_bias
        self.action_scale = action_scale
        self.basal_scaling = basal_scaling
        self.meal_announce = meal_announce
        self.meal_duration = meal_duration
        self.deterministic_meal_size = deterministic_meal_size
        self.deterministic_meal_time = deterministic_meal_time
        self.deterministic_meal_occurrence = deterministic_meal_occurrence
        self.residual_basal = residual_basal
        self.residual_bolus = residual_bolus
        self.carb_miss_prob = carb_miss_prob
        self.carb_error_std = carb_error_std
        self.residual_PID = residual_PID
        self.use_pid_load = use_pid_load
        self.fake_gt = fake_gt
        self.fake_real = fake_real
        self.suppress_carbs = suppress_carbs
        self.limited_gt = limited_gt
        self.termination_penalty = termination_penalty
        self.target = 140
        self.low_lim = 140  # Matching BB controller
        self.cooldown = 180
        self.last_cf = self.cooldown + 1
        self.start_date = start_date
        self.rolling_insulin_lim = rolling_insulin_lim
        self.rolling = []
        self.time_limit = time_limit
        if self.start_date is None:
            start_time = datetime(2018, 1, 1, 0, 0, 0)
        else:
            start_time = datetime(
                self.start_date.year,
                self.start_date.month,
                self.start_date.day,
                0,
                0,
                0,
            )
        assert bw_meals  # otherwise code wouldn't make sense
        if reset_lim is None:
            self.reset_lim = {"lower_lim": 10, "upper_lim": 1000}
        else:
            self.reset_lim = reset_lim
        self.load = load
        self.hist_init = hist_init
        self.env = None
        self.use_old_patient_env = use_old_patient_env
        self.model = model
        self.model_device = model_device
        self.use_model = use_model
        self.harrison_benedict = harrison_benedict
        self.restricted_carb = restricted_carb
        self.unrealistic = unrealistic
        self.start_time = start_time
        self.time_std = time_std
        self.weekly = weekly
        self.update_seed_on_reset = update_seed_on_reset
        self.use_custom_meal = use_custom_meal
        self.custom_meal_num = custom_meal_num
        self.custom_meal_size = custom_meal_size
        self.set_patient_dependent_values(
            patient_name
        )  # 设置 patient_name 对应的参数, 进行初始化操作
        self.env.scenario.day = 0

    def pid_load(self, n_days):
        for i in range(n_days * self.day):
            b_val = self.pid.step(self.env.CGM_hist[-1])
            act = Action(basal=0, bolus=b_val)
            _ = self.env.step(action=act, reward_fun=self.reward_fun, cho=None)

    def step(self, action):
        # 注意, 这里的 action 需要 scale, 因为需要输入 [-1, 1] 之间的值. MPC 方法不需要, 所以这里需要注意. MPC 方法使用这个环境时, 需要取消 scale.
        # action = 0.4 * action
        return self.my_step(action, cho=None)

    def translate(self, action):
        if self.action_scale == "basal":
            # 288 samples per day, bolus insulin should be 75% of insulin dose
            # split over 4 meals with 5 minute sampling rate, max unscaled value is 1+action_bias
            # https://care.diabetesjournals.org/content/34/5/1089
            action = (action + self.action_bias) * (
                (self.ideal_basal * self.basal_scaling) / (1 + self.action_bias)
            )
        else:
            action = (action + self.action_bias) * self.action_scale
        return max(0, action)

    def my_step(self, action, cho=None, use_action_scale=True):
        # cho controls if carbs are eaten, else taken from meal policy
        if type(action) is np.ndarray:
            action = action.item()
        if use_action_scale:
            if self.action_scale == "basal":
                # 288 samples per day, bolus insulin should be 75% of insulin dose
                # split over 4 meals with 5 minute sampling rate, max unscaled value is 1+action_bias
                # https://care.diabetesjournals.org/content/34/5/1089
                action = (action + self.action_bias) * (
                    (self.ideal_basal * self.basal_scaling) / (1 + self.action_bias)
                )
            else:
                action = (action + self.action_bias) * self.action_scale
        if self.residual_basal:
            action += self.ideal_basal
        if self.residual_bolus:
            ma = self.announce_meal(5)
            carbs = ma[0]
            if np.random.uniform() < self.carb_miss_prob:
                carbs = 0
            error = np.random.normal(0, self.carb_error_std)
            carbs = carbs + carbs * error
            glucose = self.env.CGM_hist[-1]
            if carbs > 0:
                carb_correct = carbs / self.CR
                hyper_correct = (
                    (glucose > self.target) * (glucose - self.target) / self.CF
                )
                hypo_correct = (
                    (glucose < self.low_lim) * (self.low_lim - glucose) / self.CF
                )
                bolus = 0
                if self.last_cf > self.cooldown:
                    bolus += hyper_correct - hypo_correct
                bolus += carb_correct
                action += bolus / 5.0
                self.last_cf = 0
            self.last_cf += 5
        if self.residual_PID:
            action += self.pid.step(self.env.CGM_hist[-1])
        if self.action_cap is not None:
            action = min(self.action_cap, action)
        if self.rolling_insulin_lim is not None:
            if np.sum(self.rolling + [action]) > self.rolling_insulin_lim:
                action = max(
                    0,
                    action
                    - (np.sum(self.rolling + [action]) - self.rolling_insulin_lim),
                )
            self.rolling.append(action)
            if len(self.rolling) > 12:
                self.rolling = self.rolling[1:]
        act = Action(basal=0, bolus=action)
        _, reward, _, info = self.env.step(act, reward_fun=self.reward_fun, cho=cho)
        # 上一步已经返回了 CGM, 但是并没有使用这个返回值 _. 而在下一步里重新计算
        state = self.get_state(self.norm)
        done = self.is_done()
        if done and self.termination_penalty is not None:
            reward = reward - self.termination_penalty
        reward = reward + self.reward_bias
        truncated = (
            False if len(self.env.CGM_hist) < 24 * 60 / 5 + self.time_limit else True
        )  # max number of steps per episode, 20 days
        # print(f"Debug simglucose_gym.py: action: {action}, state: {state}, reward: {reward}")

        if done or truncated:
            # huihang debug, print the curve before each reset
            import matplotlib.pyplot as plt

            df = pd.DataFrame()
            df["Time"] = pd.Series(self.env.time_hist)
            df["BG"] = pd.Series(self.env.BG_hist)
            df["CGM"] = pd.Series(self.env.CGM_hist)
            df["CHO"] = pd.Series(self.env.CHO_hist)
            df["insulin"] = pd.Series(self.env.insulin_hist)
            df["LBGI"] = pd.Series(self.env.LBGI_hist)
            df["HBGI"] = pd.Series(self.env.HBGI_hist)
            df["Risk"] = pd.Series(self.env.risk_hist)
            df["Magni_Risk"] = pd.Series(self.env.magni_risk_hist)

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
                f"./backup/fig/debug/RL_v3_debug_glucose_insulin.{time_start}.png"
            )
            # plt.show()
            plt.close()
        return (
            state.astype(np.float32),
            reward.astype(np.float32),
            done,
            truncated,
            info,
        )

    def announce_meal(self, meal_announce=None):
        t = (
            self.env.time.hour * 60 + self.env.time.minute
        )  # Assuming 5 minute sampling rate
        for i, m_t in enumerate(self.env.scenario.scenario["meal"]["time"]):
            # round up to nearest 5
            if m_t % 5 != 0:
                m_tr = m_t - (m_t % 5) + 5
            else:
                m_tr = m_t
            if meal_announce is None:
                ma = self.meal_announce
            else:
                ma = meal_announce
            if t < m_tr <= t + ma:
                return self.env.scenario.scenario["meal"]["amount"][i], m_tr - t
        return 0, 0

    def calculate_iob(self):
        ins = self.env.insulin_hist
        return np.dot(np.flip(self.iob, axis=0)[-len(ins) :], ins[-len(self.iob) :])

    def get_state(self, normalize=False):
        bg = self.env.CGM_hist[
            -self.state_hist :
        ]  # 如果 self.hist_init==True, 则初始长度为 self.state_hist, 否则初始为 1.
        insulin = self.env.insulin_hist[
            -self.state_hist :
        ]  # at most 288 samples, initial length is 1
        if normalize:
            bg = np.array(bg) / 400.0
            insulin = np.array(insulin) * 10
        if len(bg) < self.state_hist:  # less than 288 samples
            # fill with -1
            bg = np.concatenate((np.full(self.state_hist - len(bg), -1), bg))
        if len(insulin) < self.state_hist:
            insulin = np.concatenate(
                (np.full(self.state_hist - len(insulin), -1), insulin)
            )
        return_arr = [bg, insulin]
        if self.time:  # include time information
            time_dt = self.env.time_hist[-self.state_hist :]
            time = np.array(
                [(t.minute + 60 * t.hour) / self.sample_time for t in time_dt]
            )
            sin_time = np.sin(time * 2 * np.pi / self.day)
            cos_time = np.cos(time * 2 * np.pi / self.day)
            if normalize:
                pass  # already normalized
            if len(sin_time) < self.state_hist:
                sin_time = np.concatenate(
                    (np.full(self.state_hist - len(sin_time), -1), sin_time)
                )
            if len(cos_time) < self.state_hist:
                cos_time = np.concatenate(
                    (np.full(self.state_hist - len(cos_time), -1), cos_time)
                )
            return_arr.append(sin_time)
            return_arr.append(cos_time)
            if self.weekly:
                # binary flag signalling weekend
                if self.env.scenario.day == 5 or self.env.scenario.day == 6:
                    return_arr.append(np.full(self.state_hist, 1))
                else:
                    return_arr.append(np.full(self.state_hist, 0))
        if self.meal:  # include meal information
            cho = self.env.CHO_hist[-self.state_hist :]
            if normalize:
                cho = np.array(cho) / 20.0
            if len(cho) < self.state_hist:
                cho = np.concatenate((np.full(self.state_hist - len(cho), -1), cho))
            return_arr.append(cho)
        if self.meal_announce is not None:
            meal_val, meal_time = self.announce_meal()
            future_cho = np.full(self.state_hist, meal_val)
            return_arr.append(future_cho)
            future_time = np.full(self.state_hist, meal_time)
            return_arr.append(future_time)
        if self.fake_real:
            state = self.env.patient.state
            return np.stack([state for _ in range(self.state_hist)]).T.flatten()
        if self.gt:
            if self.fake_gt:
                iob = self.calculate_iob()
                cgm = self.env.CGM_hist[-1]
                if normalize:
                    state = np.array([cgm / 400.0, iob * 10])
                else:
                    state = np.array([cgm, iob])
            else:
                state = self.env.patient.state
            if self.meal_announce is not None:
                meal_val, meal_time = self.announce_meal()
                state = np.concatenate((state, np.array([meal_val, meal_time])))
            if normalize:
                # just the average of 2 days of adult#001, these values are patient-specific
                norm_arr = np.array(
                    [
                        4.86688301e03,
                        4.95825609e03,
                        2.52219425e03,
                        2.73376341e02,
                        1.56207049e02,
                        9.72051746e00,
                        7.65293763e01,
                        1.76808549e02,
                        1.76634852e02,
                        5.66410518e00,
                        1.28448645e02,
                        2.49195394e02,
                        2.73250649e02,
                        7.70883882e00,
                        1.63778163e00,
                    ]
                )
                if self.meal_announce is not None:
                    state = state / norm_arr
                else:
                    state = state / norm_arr[:-2]
            if self.suppress_carbs:
                state[:3] = 0.0
            if self.limited_gt:
                state = np.array([state[3], self.calculate_iob()])
            return state.astype(np.float32)

        # 默认情况下. 返回值是一个向量, 包含了 长度为 self.state_hist 的 BG 与同样长度的 insulin. 先 BG, 再 insulin.
        # 其他的信息, 可以通过 self.time, self.meal, self.meal_announce, self.fake_real, self.gt 调整
        return np.stack(return_arr).flatten()

    def avg_risk(self):
        return np.mean(self.env.risk_hist[max(self.state_hist, 288) :])

    def avg_magni_risk(self):
        return np.mean(self.env.magni_risk_hist[max(self.state_hist, 288) :])

    def glycemic_report(self):
        bg = np.array(self.env.BG_hist[max(self.state_hist, 288) :])
        ins = np.array(self.env.insulin_hist[max(self.state_hist, 288) :])
        hypo = (bg < 70).sum() / len(bg)
        hyper = (bg > 180).sum() / len(bg)
        euglycemic = 1 - (hypo + hyper)
        return bg, euglycemic, hypo, hyper, ins

    def is_done(self):
        # print(f"Debug simglucose_gym.py is_done(): BG_hist[-1]: {self.env.BG_hist[-1] < self.reset_lim['lower_lim']}, reset_lim: {self.env.BG_hist[-1] > self.reset_lim['upper_lim']}")
        return (
            self.env.BG_hist[-1] < self.reset_lim["lower_lim"]
            or self.env.BG_hist[-1] > self.reset_lim["upper_lim"]
        )

    def increment_seed(self, incr=1):
        self.seeds.numpy_seed += incr
        self.seeds.scenario_seed += incr
        self.seeds.sensor_seed += incr

    def reset(self, seed=None, return_info=None, options=None):
        # tmp_info = False
        return self.my_reset()

    def set_patient_dependent_values(self, patient_name):
        self.patient_name = patient_name
        vpatient_params = pd.read_csv(self.patient_para_file)
        quest = pd.read_csv(self.control_quest)
        self.kind = self.patient_name.split("#")[0]
        self.bw = vpatient_params.query('Name=="{}"'.format(self.patient_name))[
            "BW"
        ].item()
        self.u2ss = vpatient_params.query('Name=="{}"'.format(self.patient_name))[
            "u2ss"
        ].item()
        self.ideal_basal = self.bw * self.u2ss / 6000.0
        self.CR = quest.query('Name=="{}"'.format(patient_name)).CR.item()
        self.CF = quest.query('Name=="{}"'.format(patient_name)).CF.item()
        if self.rolling_insulin_lim is not None:
            self.rolling_insulin_lim = (
                (self.rolling_insulin_lim * self.bw)
                / self.CR
                * self.rolling_insulin_lim
            ) / 5
        else:
            self.rolling_insulin_lim = None
        iob_all = joblib.load("{}/iob.pkl".format(self.pid_env_path))
        self.iob = iob_all[self.patient_name]
        pid_df = pd.read_csv(self.pid_para_file)
        if patient_name not in pid_df.name.values:
            raise ValueError("{} not in PID csv".format(patient_name))
        pid_params = pid_df.loc[pid_df.name == patient_name].squeeze()
        self.pid = pid.PID(
            setpoint=pid_params.setpoint,
            kp=pid_params.kp,
            ki=pid_params.ki,
            kd=pid_params.kd,
        )
        patient = T1DPatientNew.withName(patient_name, self.patient_para_file)
        sensor = CGMSensor.withName(
            "Dexcom", self.sensor_para_file, seed=self.seeds.sensor_seed
        )
        if self.time_std is None:
            scenario = RandomBalancedScenario(
                bw=self.bw,
                start_time=self.start_time,
                seed=self.seeds.scenario_seed,
                kind=self.kind,
                restricted=self.restricted_carb,
                harrison_benedict=self.harrison_benedict,
                unrealistic=self.unrealistic,
                deterministic_meal_size=self.deterministic_meal_size,
                deterministic_meal_time=self.deterministic_meal_time,
                deterministic_meal_occurrence=self.deterministic_meal_occurrence,
                meal_duration=self.meal_duration,
            )
        elif self.use_custom_meal:
            scenario = CustomBalancedScenario(
                bw=self.bw,
                start_time=self.start_time,
                seed=self.seeds.scenario_seed,
                num_meals=self.custom_meal_num,
                size_mult=self.custom_meal_size,
            )
        else:
            scenario = SemiRandomBalancedScenario(
                bw=self.bw,
                start_time=self.start_time,
                seed=self.seeds.scenario_seed,
                time_std_multiplier=self.time_std,
                kind=self.kind,
                harrison_benedict=self.harrison_benedict,
                meal_duration=self.meal_duration,
            )
        pump = InsulinPump.withName("Insulet", self.insulin_pump_para_file)
        self.env = T1DSimEnv(
            patient=patient,
            sensor=sensor,
            pump=pump,
            scenario=scenario,
            sample_time=self.sample_time,
            source_dir=self.source_dir,
        )
        if self.hist_init:
            self.env_init_dict = joblib.load(
                "{}/{}_data.pkl".format(self.pid_env_path, self.patient_name)
            )  # load patient data for initialization
            self.env_init_dict["magni_risk_hist"] = []
            for bg in self.env_init_dict["bg_hist"]:  # length 298
                self.env_init_dict["magni_risk_hist"].append(magni_risk_index([bg]))
            self.my_hist_init()

    def my_reset(self, seed=None, return_info=False, options=None):
        if self.update_seed_on_reset:
            self.increment_seed()
        if self.use_model:
            if self.load:
                self.env = joblib.load(
                    "{}/{}_fenv.pkl".format(self.pid_env_path, self.patient_name)
                )
                self.env.model = self.model
                self.env.model_device = self.model_device
                self.env.norm_params = self.norm_params
                self.env.state = self.env.patient.state
                self.env.scenario.kind = self.kind
            else:
                self.env.reset()
        else:
            if self.load:
                if self.use_old_patient_env:
                    self.env = joblib.load(
                        "{}/{}_env.pkl".format(self.pid_env_path, self.patient_name)
                    )
                    self.env.model = None
                    self.env.scenario.kind = self.kind
                else:
                    self.env = joblib.load(
                        "{}/{}_fenv.pkl".format(self.pid_env_path, self.patient_name)
                    )
                    self.env.model = None
                    self.env.scenario.kind = self.kind
                if self.time_std is not None:
                    self.env.scenario = SemiRandomBalancedScenario(
                        bw=self.bw,
                        start_time=self.start_time,
                        seed=self.seeds.scenario_seed,
                        time_std_multiplier=self.time_std,
                        kind=self.kind,
                        harrison_benedict=self.harrison_benedict,
                        meal_duration=self.meal_duration,
                    )
                self.env.sensor.seed = self.seeds.sensor_seed
                self.env.scenario.seed = self.seeds.scenario_seed
                self.env.scenario.day = 0
                self.env.scenario.weekly = self.weekly
                self.env.scenario.kind = self.kind
            else:
                if self.universal:
                    patient_name = np.random.choice(self.universe)
                    self.set_patient_dependent_values(patient_name)
                self.env.sensor.seed = self.seeds.sensor_seed
                self.env.scenario.seed = self.seeds.scenario_seed
                self.env.reset()
                self.pid.reset()
                if self.use_pid_load:
                    self.pid_load(1)
                if self.hist_init:
                    self.my_hist_init()

        #
        info = {}

        return self.get_state(self.norm), info

    def my_hist_init(self):
        self.rolling = []
        env_init_dict = copy.deepcopy(self.env_init_dict)
        self.env.patient._state = env_init_dict["state"]
        self.env.patient._t = env_init_dict["time"]
        if self.start_date is not None:
            # need to reset date in start time
            orig_start_time = env_init_dict["time_hist"][0]
            new_start_time = datetime(
                year=self.start_date.year,
                month=self.start_date.month,
                day=self.start_date.day,
            )
            new_time_hist = (
                (np.array(env_init_dict["time_hist"]) - orig_start_time)
                + new_start_time
            ).tolist()
            self.env.time_hist = new_time_hist
        else:
            self.env.time_hist = env_init_dict["time_hist"]
        self.env.BG_hist = env_init_dict["bg_hist"]
        self.env.CGM_hist = env_init_dict["cgm_hist"]
        self.env.risk_hist = env_init_dict["risk_hist"]
        self.env.LBGI_hist = env_init_dict["lbgi_hist"]
        self.env.HBGI_hist = env_init_dict["hbgi_hist"]
        self.env.CHO_hist = env_init_dict["cho_hist"]
        self.env.insulin_hist = env_init_dict["insulin_hist"]
        self.env.magni_risk_hist = env_init_dict["magni_risk_hist"]

    def my_seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This is manually seeded to avoid issues
        seed1 = seed1 % 2**31
        seed2 = (seed1 + 2024) % 2**31
        seed3 = (seed2 + 2024) % 2**31

        # huihang debug
        if False:
            return [seed1, seed2, seed3]
        else:
            # print(f"Modify simglucose_gym.py my_seed()")
            return [2024, 2024, 2024]

    @property
    def action_space(self):
        # return spaces.Box(low=0, high=0.1, shape=(1,))
        return spaces.Box(low=-1.0, high=1.0, shape=(1,))

    @property
    def observation_space(self):
        st = self.get_state()  # np.array, length 576
        if self.gt:  # if Oracle
            return spaces.Box(low=0, high=np.inf, shape=(len(st),))
        else:  # if not Oracle
            num_channels = int(len(st) / self.state_hist)
            return spaces.Box(
                low=0, high=np.inf, shape=(num_channels * self.state_hist,)
            )

        # huihang debug
        # return spaces.Box(low=np.array([0.0]*6), high=np.array([np.inf]*6), dtype=np.float32)


# %%
if __name__ == "__main__":
    from BGEnv.UVAPEnv.envs import reward_functions

    def reward_name_to_function(reward_name):

        if reward_name == "risk_diff":
            reward_fun = reward_functions.risk_diff
        elif reward_name == "risk_diff_bg":
            reward_fun = reward_functions.risk_diff_bg
        elif reward_name == "risk":
            reward_fun = reward_functions.reward_risk
        elif reward_name == "risk_bg":
            reward_fun = reward_functions.risk_bg
        elif reward_name == "magni_bg":
            reward_fun = reward_functions.magni_reward
        elif reward_name == "cameron_bg":
            reward_fun = reward_functions.cameron_reward
        elif reward_name == "eps_risk":
            reward_fun = reward_functions.epsilon_risk
        elif reward_name == "target_bg":
            reward_fun = reward_functions.reward_target
        elif reward_name == "cgm_high":
            reward_fun = reward_functions.reward_cgm_high
        elif reward_name == "bg_high":
            reward_fun = reward_functions.reward_bg_high
        elif reward_name == "cgm_low":
            reward_fun = reward_functions.reward_cgm_low
        else:
            raise ValueError("{} not a proper reward_name".format(reward_name))
        return reward_fun

    reward_func = reward_name_to_function("magni_bg")
    env = DeepSACT1DEnv(
        reward_fun=reward_func,
        bw_meals=True,
        n_hours=4,
        hist_init=True,
        source_dir="/Users/huihang/Library/CloudStorage/OneDrive-Personal/Repository/RL-Control/code/",
    )
