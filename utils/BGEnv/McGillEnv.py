#  %%
'''
In this file, we define the environment for the diabetes management problem base on the OpenAI Gym interface and McGill's simulator.
version: 2024-03-23
author: huihang@mail.ustc.edu.cn
'''

# %% 
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import gymnasium as gym
from gymnasium import spaces
from scipy.optimize import minimize # for MPC

# %% Environment for both RL and MPC
class DiabetesEnv(gym.Env):
  metadata = {'render.modes': ['console'], 'render_fps': 24*60}

  def __init__(self, time_horizon=24*60, sampling_interval=5, obs_len=3, flag_rl=True):
    """
    Initializes the environment.
    Parameters:
    - time_horizon: The time horizon for the simulation. In minutes. Default is 24*60 (min).
    - sampling_interval: The sampling interval for measurements of glucose level. Default is 5 (min).
    """
    super(DiabetesEnv, self).__init__()
    # state space: previous 3 glucose levels
    self.obs_len = obs_len
    self.observation_space = spaces.Box(low=0.0, high=600.0, shape=(self.obs_len*2,), dtype=np.float32)

    self.flag_init_hist = True  # False, 这是一个迭代的定义, 因为在初始化时并没有历史数据, 所以在初始化过程中要设置为 False, 否则会出错
    self.len_hist = 4*60        # 4 hours
    self.flag_rl = flag_rl    # if this is rl method, scale the action
    
    # action space: insulin dose rate
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    # Time step counter
    self.current_step = 0
    
    # Define the simulation parameters
    self.G0 = 6.0*18  # Initial glucose level (mg/dL)
    self.nt = time_horizon  # time for the simulation, horizon, in minutes

    # Parameters in McGill's simulator
    self.Qb = 0.0   # Basal insulin infusion rate
    self.CHO = [30.0, 40.0, 30.0] # Carbohydrate content of meals, [breakfast, lunch, dinner]
    self.para_ctrl = {
      'start': 7.0, # Start time of the experiment (h)
      'dt': 1.0,    # Time interval for simulation (min)
      'st': sampling_interval,    # Sampling interval for measurements glucose level (mins)
      'tr': 10.0,   # Time interval for diurnal effects to jump to the next value (min)
      'ti': 5.0     # Time interval for control action (min)
    }
    self.para_subj = {
      'w': 80, # Weight of the subject
      # Parameters for glucose dynamics
      'pi1': 0.869016895, 
      'kis1': 0.021952527, 
      'kis2': 0.015947002, 
      'kif': 0.0646566, 
      'ke': 0.08127905, 
      'ci': 0.000745756, 
      'thetai': 0.952513248, 
      'sigmai': 0.004254322, 
      # Parameters for insulin dynamics
      'sigma1': 5.632192116, 
      'ka1': 0.067700384, 
      'St': 0.003587876, 
      'ka2': 0.167543264, 
      'Sd': 0.000157999, 
      'ka3': 0.26798852, 
      'Se': 0.086717971, 
      'F01': 5.972762422, 
      'k12': 0.164483856, 
      'EGP0': 26.8076167, 
      'km': 0.050053418, 
      'd': 12.0252328, 
      'pm': 0.042901359, 
      # Parameters for diurnal effects
      'thetam': 0.948236952, 
      'sigmam': 0.005021698, 
      'thetag': 0.946816086, 
      'sigmag': 0.501441709, 
      'sigma2': 0.361508092 
    }
    self.Vi = 190 # Insulin distribution volume, given by the paper
    self.V = 160 # Volume of distribution of glucose, given by the paper
    # subject-specific parameters
    self.w = self.para_subj['w']
    # user-specified control parameters
    self.start = self.para_ctrl['start']
    self.dt = self.para_ctrl['dt']
    self.st = self.para_ctrl['st']
    self.tr = self.para_ctrl['tr']
    self.ti = self.para_ctrl['ti']
    # Dictionary with 'lower_lim' and 'upper_lim' keys, representing the lower and upper limits for the glucose level. Default is None.
    self.reset_lim = {'lower_lim': 10, 'upper_lim': 1000}
    self.terminate_penalty = 1e5  # penalty for terminating the episode early (glucose level out of safty bounds)

    # Time course
    self.n = int(self.nt) # Total time course (in minutes)
    self.tseq = np.arange(self.n) # Time sequence (actually in steps, not in minutes)

    # History of glucse levels (need rest for new episode)
    self.observed_glucose_levels = [self.G0] # Observed glucose levels
    self.blood_glucose_levels = [] # Blood glucose levels
    self.insulin_doses = [] # Insulin doses

    # initialize the trajectory
    self.reset()

  @staticmethod
  def diurnal_func(n=100, tr=10, para=None):
    """
    Models diurnal effect of a virtual T1D subject within McGill simulator.
    
    Parameters:
    - n: length of time course (in min) of an experiment.
    - tr: time interval (min) that diurnal effects jump to the next value.
    - para: Dictionary with subject-specific parameters.
    
    Returns:
    - Dictionary with 'Im', 'fg', 'fm' representing diurnal effects which means insulin sensitivity, glucose effectiveness, and meal effect, respectively.
    """
    if para is None:
      para = {
        'thetai': 0.0, 
        'thetag': 0.0, 
        'thetam': 0.0, 
        'sigmai': 1.0, 
        'sigmag': 1.0, 
        'sigmam': 1.0
      }

    # Initial values
    Im0, fg0, fm0 = 1.0, 0.0, 1.0

    # Time course
    tseq = np.arange(1, n + 1)

    # Steps and random walks
    steps = int(np.ceil(n / tr)) + 1
    error = np.random.normal(0, [para['sigmai'], para['sigmag'], para['sigmam']], size=(steps, 3))

    # Define functions Im, fg, fm
    Im = np.full(steps, Im0)
    fg = np.full(steps, fg0)
    fm = np.full(steps, fm0)

    for s in range(1, steps):
      Im[s] = np.exp(para['thetai'] * np.log(Im[s-1]) + error[s-1, 0])
    
      fg[s] = para['thetag'] * fg[s-1] + error[s-1, 1]
      fm[s] = np.exp(para['thetam'] * np.log(fm[s-1]) + error[s-1, 2])
    
    # Interpolate piece-wise linear functions
    t_new = np.arange(0, steps)
    Im = interp1d(t_new * tr + 1, Im, kind='linear', fill_value="extrapolate")(tseq)
    fg = interp1d(t_new * tr + 1, fg, kind='linear', fill_value="extrapolate")(tseq)
    fm = interp1d(t_new * tr + 1, fm, kind='linear', fill_value="extrapolate")(tseq)
    
    return {'Im': Im, 'fg': fg, 'fm': fm}

  @staticmethod
  def mealplan_func(time=[8, 12, 18], CHO=[30, 40, 30], sd=[5, 5, 5], during=3):
    """
    Specifies the meals a subject takes during the experiment.
    
    Parameters:
    - time: Times of meals in a day in 24h format.
    - CHO: Expected carbohydrate content of meals.
    - sd: Standard deviation of carbohydrate content.
    - during: Duration of the experiment in days.
    
    Returns:
    - DataFrame with meal times and CHO for the experiment duration.
    """
    dd = np.ceil(during).astype(int)
    
    CHO = np.random.multivariate_normal(mean=CHO, cov=np.diag(sd), size=dd).ravel() # this is ravel row-by-row, in order of day

    meal_times = []
    for d in range(dd):
      meal_times.append(np.array(time) + d * 24)
    meal_times = np.array(meal_times).ravel()

    mealplan = {'time': meal_times[meal_times < during * 24], 'CHO': CHO[meal_times < during * 24]}
    return mealplan

  @staticmethod
  def diurnal_target(y, daytime):
    """
    Adjusts the target glucose level based on the time of day.
    - Z = 0 if y is within the target range, 
    - Z = y - lower if y < lower
    - Z = y - upper if y > upper.
    Parameters:
    - y: Current glucose level.
    - daytime: Current time of day.
    
    Returns:
    - Adjusted glucose level. 
    """
    upper = 140.0
    lower = 80.0 if 8.0 <= daytime <= 22.0 else 110.0
    if y - lower >= 0.0 and y - upper >= 0.0:
      Z = y - upper
    elif y - upper <= 0.0 and y - lower >= 0.0:
      Z = 0
    else:
      Z = y - lower
    return Z

  @staticmethod
  def reward(glucose_level_, insulin_dose_=None, daytime_=None):
    """
    Reward function for the environment.
    Parameters:
    - blood_glouse_: Current glucose level.
    - insulin_dose_: Current insulin dose.
    - daytime_: Current time of day.
    
    Returns:
    - Reward value.
    """

    # reward function: 
    # 1. use diurnal_target(): if glucose level is in [80,140] reward is higher, else negative reward
    # 2. use the difference from a target value (120) as reward
    # 3. additional penalize insulin dose (not good 所有的 action 都是 0)

    reward = 0.0

    flag = 'fox2020pmlr'

    if flag == 'interval':
      upper = 140.0
      lower = 80.0 if ((8.0 <= daytime_) and (daytime_ <= 22.0)) else 110.0
      if glucose_level_ - upper >= 0.0:
        reward = -10 # (upper - glucose_level) / 2
      elif glucose_level_ - upper < 0.0 and glucose_level_ - lower >= 0.0:
        reward = 100.0 - np.abs(glucose_level_ - 120.0)
      else:
        reward = -20 # glucose_level - lower
      reward -= 76 * insulin_dose_
    elif flag == 'fox2020pmlr':
      c0, c1, c2 = 3.35506, 0.8353, 3.7932
      reward = - 10 * (c0 * (np.log(np.clip(glucose_level_, a_min=1, a_max=1000))**c1 - c2))**2
    return reward

  def MPC_func(self, Gts, Nu, Np, ui0, daytime):
    """
    Specifies the insulin infusion rate via the MPC controller.
    
    Parameters:
    - Gts: Glucose measurements at time t-2, t-1, t.
    - Nu: Control horizon.
    - Np: Prediction horizon.
    - ui0: Initial value of insulin infusion rate ui(t).
    - daytime: Current daytime.
    
    Returns: 
      A dictionary with keys 
        - 'MPC.u' (optimal control input)
        - 'x' (predicted state vector).
    """
    # Gts = Gts # unit mg/dL
    DIR = 30  # subject’s daily insulin requirement (units per kg per day)
    
    # Initialize state vector based on glucose measurements
    x_vec = np.array([Gts[2], Gts[1], Gts[0]]) # reverse order ??, x is the glucose level at time t-2, t-1, t
    # MPC parameters
    p1, p2 = 0.98, 0.965
    A = np.array([
      [p1 + 2 * p2, -2 * p1 * p2 - p2**2, p1 * p2**2], 
      [1, 0, 0], 
      [0, 1, 0]
    ])
    K = 90 * (p1 - 1) * (p2 - 1)**2
    B = np.array([1800 * K / DIR, 0, 0])
    Cv = np.array([0.1, 0, -0.1])
    D_hat = 1000 # D_hat is the weight for the control input used in the cost function
    v = np.dot(Cv, x_vec) # v is the predicted rate of change of glucose level
    eps = 0.1
    R1, R2 = 6500, 100
    
    # Cost function for MPC optimization
    def J_cost_MPC(u_vec):
      J_MPC = 0
      x_vec_local = np.copy(x_vec)
      zk = self.diurnal_target(x_vec_local[2], daytime)
      v_local = v
      for k in range(Np): # Prediction horizon Np
        if v_local >= 0:
          Q_k = 1
        elif v_local <= -1:
          Q_k = eps
        else:
          Q_k = 0.5 * (np.cos(v_local * np.pi) * (1 - eps) + 1 + eps)
        
        if zk >= 0:
          c = Q_k
        else:
          c = 1
        v_hat = max(v_local, 0)
        # Cost function: J = c * zk^2 + D_hat * v_hat^2, where zk is the loss of glucose level, v_hat is the predicted rate of change of glucose level
        J_MPC += c * (zk)**2 + D_hat * v_hat**2  
        if k < Nu:
          R = R1 if u_vec[k] > 0 else R2
          J_MPC += R * (u_vec[k])**2
          x_vec_local = np.dot(A, x_vec_local) + B * u_vec[k]  # Update state
        else:
          x_vec_local = np.dot(A, x_vec_local)  # uk = 0 for k > Nu
        
        # Update zk and v for the next iteration
        zk = self.diurnal_target(x_vec_local[2], daytime)  # x_vec_local[2] is the predicted glucose level
        v_local = np.dot(Cv, x_vec_local)
      
      return J_MPC
    
    # Optimization to find the best control inputs
    x_trace = []
    u_initial = np.ones(Nu) * 1  # Initial guess for control inputs
    bounds = [(-ui0, 0.4)] * Nu  # Bounds for control inputs
    
    for k in range(Np):
      # predict the glucose level in the next Np steps
      if k < Nu:
        out_OP = minimize(J_cost_MPC, u_initial, bounds=bounds, method='L-BFGS-B')
        u_opt = out_OP.x[0] # the first element in U is set as the current control variable
        x_vec = np.dot(A, x_vec) + B * u_opt # Update state vector with the optimal control input (predicted glucose level)
      else:
        x_vec = np.dot(A, x_vec) # Update state vector without control input (predicted glucose level)
      x_trace.append(x_vec)
    
    return {'MPC.u': u_opt+ui0, 'x': x_vec}

  def is_done(self):
    return self.observed_glucose_levels[-1] < self.reset_lim['lower_lim'] or self.observed_glucose_levels[-1] > self.reset_lim['upper_lim']

  def step(self, action):
    '''
    The agent takes a step in the environment.
    Parameters:
      - action 1-dim np.array: The action taken by the agent.
    '''
    # initialize the return values
    done = truncated = 0

    if action is None: 
      action = np.array([self.f['ui'][0]], dtype=np.float32)
    elif self.flag_rl:
      # action = np.clip(action * 0.4, a_min=0, a_max=1.0, dtype=np.float32)
      # if this is for MPC, no need to scale
      action = np.clip(action*0.4, a_min=0, a_max=1.0, dtype=np.float32)
    insulin_dose = action[0]
    while True:  
      t = self.current_step
      daytime = (self.start + t / 60) % 24 # step to day-time by 24 hours
      self.f['ui'][t] = insulin_dose
      # dynamics; in reality, use diurnal_func, mealplan_func here
      self.dx.loc[t, 'Qis1'] = self.f['ui'][t] * self.para_subj['pi1'] - self.x['Qis1'][t] * self.para_subj['kis1']
      self.dx.loc[t, 'Qis2'] = self.x['Qis1'][t] * self.para_subj['kis1'] - self.x['Qis2'][t] * self.para_subj['kis2']
      self.dx.loc[t, 'Qif1'] = self.f['ui'][t] * (1 - self.para_subj['pi1']) - self.x['Qif1'][t] * self.para_subj['kif']
      self.dx.loc[t, 'Qif2'] = self.x['Qif1'][t] * self.para_subj['kif'] - self.x['Qif2'][t] * self.para_subj['kif']

      # Plasma Insulin Kinetics Subsystem
      self.dx.loc[t, 'Qi'] = (self.x['Qis2'][t] * self.para_subj['kis2'] + self.x['Qif2'][t] * self.para_subj['kif']) * self.f['Im'][t] - self.x['Qi'][t] * self.para_subj['ke'] + self.para_subj['ci']
      self.z.loc[t, 'Ip'] = self.x['Qi'][t] / (self.Vi * self.w) * 1e6

      # Insulin Action Subsystem
      self.dx.loc[t, 'x1'] = - self.para_subj['ka1'] * self.x['x1'][t] + self.para_subj['ka1'] * self.para_subj['St'] * self.z['Ip'][t]
      self.dx.loc[t, 'x2'] = - self.para_subj['ka2'] * self.x['x2'][t] + self.para_subj['ka2'] * self.para_subj['Sd'] * self.z['Ip'][t]
      self.dx.loc[t, 'x3'] = - self.para_subj['ka3'] * self.x['x3'][t] + self.para_subj['ka3'] * self.para_subj['Se'] * self.z['Ip'][t]

      # Gut Glucose Absorption Subsystem
      tU = np.maximum(t - (self.mealplan['time'] - self.start) * 60, 0) # time since meal, in minutes
      self.z.loc[t, 'Um1'] = np.sum(self.para_subj['km']**2 * tU * np.exp(-self.para_subj['km'] * tU) * self.mealplan['CHO'] * 5551 / self.w * self.para_subj['pm'])
      self.z.loc[t, 'Um2'] = np.sum(np.where(tU > self.para_subj['d'], self.para_subj['km']**2 * (tU - self.para_subj['d']) * np.exp(-self.para_subj['km'] * (tU - self.para_subj['d'])) * self.mealplan['CHO'] * 5551 / self.w * (1 - self.para_subj['pm']), 0))
      self.z.loc[t, 'Um'] = (self.z['Um1'][t] + self.z['Um2'][t]) * self.f['fm'][t]

      # Glucose Kinetics Subsystem
      EGP = np.where(self.x['x3'][t] < 1, self.para_subj['EGP0'] * (1 - self.x['x3'][t]), 0)
      self.dx.loc[t, 'Q1'] = - self.para_subj['F01'] * self.x['Q1'][t] / self.V / (1 + self.x['Q1'][t] / self.V) - self.x['x1'][t] * self.x['Q1'][t] + self.para_subj['k12'] * self.x['Q2'][t] + EGP + self.f['fg'][t] + self.z['Um'][t]
      self.dx.loc[t, 'Q2'] = self.x['x1'][t] * self.x['Q1'][t] - (self.para_subj['k12'] + self.x['x2'][t]) * self.x['Q2'][t]
      self.z.loc[t, 'G'] = self.x['Q1'][t] / self.V
      glucose_level = self.z['G'][t] * 18 # convert from mmol/L to mg/dL
      self.blood_glucose_levels.append(glucose_level)

      # Update state
      if t < self.n - 1: self.x.iloc[t+1] = self.x.iloc[t] + self.dx.iloc[t] * self.dt
    
      # Increment time step
      self.current_step += 1
      done = self.is_done()
      truncated = (self.current_step >= self.n)
      if (self.current_step % self.st == 0) or done or truncated: break

    observed_glucose_level = glucose_level
    self.observed_glucose_levels.append(observed_glucose_level) # keep the last observed glucose level
    self.insulin_doses.append(insulin_dose)

    # update the state
    tmp_state_bg = np.clip(self.observed_glucose_levels[-self.obs_len:], a_min=0, a_max=600.0, dtype=np.float32)
    if len(tmp_state_bg) < self.obs_len:
      if self.flag_init_hist:
        tmp_state_bg = np.concatenate([np.array([self.history['observed_glucose_levels'][-(self.obs_len-len(tmp_state_bg)):]], dtype=np.float32).flatten(), tmp_state_bg])
      else:
        tmp_state_bg = np.concatenate([np.array([self.G0]*(self.obs_len-len(tmp_state_bg)), dtype=np.float32), tmp_state_bg])
    tmp_state_insulin = self.insulin_doses[-self.obs_len:]
    if len(tmp_state_insulin) < self.obs_len:
      if self.flag_init_hist:
        tmp_state_insulin = np.concatenate([np.array([self.history['insulin_rates'][-(self.obs_len-len(tmp_state_insulin)):]], dtype=np.float32).flatten(), tmp_state_insulin])
      else: 
        tmp_state_insulin = np.concatenate([np.array([self.ui0]*(self.obs_len-len(tmp_state_insulin)), dtype=np.float32), tmp_state_insulin])
    self.state = np.concatenate([tmp_state_bg, tmp_state_insulin], dtype=np.float32)

    # obtain the reward
    reward = self.reward(observed_glucose_level, insulin_dose, daytime)

    # Check if the simulation ends
    if done:
      reward -= self.terminate_penalty

    # debug
    if reward is None or reward < -2e5:
      print(f"reward: {reward}, glucose_level: {observed_glucose_level}, insulin_dose: {insulin_dose}, daytime: {daytime}")
      raise ValueError("reward is None or too low")
    
    # Ensure 'info' dictionary is returned, even if empty
    info = {}

    return self.state, reward, done, truncated, info

  def reset(self, seed=None, return_info=False, options=None):

    # Initialization of state variables and derivatives
    namesvar = ['Qis1', 'Qis2', 'Qif1', 'Qif2', 'Qi', 'x1', 'x2', 'x3', 'Q1', 'Q2']
    # 'Qis1', 'Qis2', 'Qif1', 'Qif2' are the subcutaneous insulin absorption subsystem, 'Qi' is the plasma insulin concentration, 'x1', 'x2', 'x3' are the insulin action subsystem, 'Q1', 'Q2' are glucose mass in the central and peripheral compartments
    # create a dataframe with namesvar as columns and n rows
    self.x = pd.DataFrame(np.zeros((self.n, len(namesvar))), columns=namesvar)
    self.dx = self.x.copy() # Derivatives
    # State variables for glucose and insulin dynamics, for intermediate quantities not involved in ODEs. 'Ip' is the plasma insulin concentration, 'G' is the plasma glucose concentration, 'Um' is the meal effect, 'Um1' and 'Um2' are the two components of the meal effect.
    namesintv = ["Ip", "G", "Um", "Um1", "Um2"]
    self.z = pd.DataFrame(np.zeros((self.n, len(namesintv))), columns=namesintv)

    # Solve initial value for Ip0 via quadratic function
    a = self.para_subj['Sd'] * (self.para_subj['Se'] * self.para_subj['EGP0'] + self.para_subj['St'] * self.G0 * self.V / 18)
    m1 = self.para_subj['F01'] * self.G0 / 18 / (1 + self.G0 / 18) - self.para_subj['EGP0']
    b = m1 * self.para_subj['Sd'] + self.para_subj['k12'] * self.para_subj['Se'] * self.para_subj['EGP0']
    cc = m1 * self.para_subj['k12']
    self.z.loc[0, 'Ip'] = (np.sqrt(b ** 2 - 4 * a * cc) - b) / (2 * a)

    # Initial insulin infusion rate ui0
    ui0 = self.z['Ip'][0] * self.w * self.Vi * self.para_subj['ke'] * 1e-6 - self.para_subj['ci']
    self.ui0 = ui0
    # Diurnal effect
    self.f = self.diurnal_func(n=self.n, 
                    tr=self.para_ctrl['tr'], 
                    para={
                      'thetai': self.para_subj['thetai'],
                      'thetag': self.para_subj['thetag'],
                      'thetam': self.para_subj['thetam'],
                      'sigmai': self.para_subj['sigmai'],
                      'sigmag': self.para_subj['sigmag'],
                      'sigmam': self.para_subj['sigmam']
                    })

    self.f['ui'] = np.repeat(ui0, self.n) # Insulin infusion rate

    # solve the rest of initial values
    self.x.loc[0, 'Qis1'] = self.f['ui'][0] * self.para_subj['pi1'] / self.para_subj['kis1']
    self.x.loc[0, 'Qis2'] = self.x['Qis1'][0] * self.para_subj['kis1'] / self.para_subj['kis2'] + self.Qb * self.para_subj['pi1']
    self.x.loc[0, 'Qif1'] = self.f['ui'][0] * (1 - self.para_subj['pi1']) / self.para_subj['kif']
    self.x.loc[0, 'Qif2'] = self.x['Qif1'][0] + self.Qb * (1 - self.para_subj['pi1'])
    self.x.loc[0, 'Qi'] = self.z['Ip'][0] * self.Vi * self.w * 1e-6

    self.x.loc[0, 'x1'] = self.para_subj['St'] * self.z['Ip'][0]
    self.x.loc[0, 'x2'] = self.para_subj['Sd'] * self.z['Ip'][0]
    self.x.loc[0, 'x3'] = self.para_subj['Se'] * self.z['Ip'][0]
    self.x.loc[0, 'Q1'] = self.G0 * self.V / 18
    self.x.loc[0, 'Q2'] = self.x['Q1'][0] * self.x['x1'][0] / (self.x['x2'][0] + self.para_subj['k12'])
    self.z.loc[0, 'Um'] = self.z.loc[0, 'Um1'] = self.z.loc[0, 'Um2'] = 0.0
    self.z.loc[0, 'G'] = self.G0 / 18
    self.mealplan = self.mealplan_func(CHO=self.CHO, during=self.n / 60 / 24)

    # Reset the state to initial conditions
    # if use history
    if not self.flag_init_hist: 
      self.state = np.concatenate([[self.G0]*3, [self.ui0]*3], dtype=np.float32)
      self.current_step = 0
      self.observed_glucose_levels = [self.G0] # Observed glucose levels
      self.blood_glucose_levels = [] # Blood glucose levels
      self.insulin_doses = [] # Insulin doses
    else: 
      self.current_step = 0
      self.observed_glucose_levels = [self.G0] # Observed glucose levels
      self.blood_glucose_levels = [] # Blood glucose levels
      self.insulin_doses = [] # Insulin doses
      self.flag_init_hist = False
      self.init_history()
      self.flag_init_hist = True
      self.current_step = 0
      self.state = np.array(self.history['observed_glucose_levels'][-self.obs_len:] + self.history['insulin_rates'][-self.obs_len:], dtype=np.float32)
      self.observed_glucose_levels = [self.G0] # Observed glucose levels
      self.blood_glucose_levels = [] # Blood glucose levels
      self.insulin_doses = [] # Insulin doses

    # place holder for reset() return values
    info = {}

    return self.state, info

  def init_history(self):
    # Initialize the history of glucose levels and insulin doses in the past 4 hours (length is the same as the observation space)
    # using MPC method. 

    print("[Init Hist] Initializing the history of glucose levels and insulin doses in the past 4 hours.")
    
    len_hist = int(self.len_hist / self.st) # 24 hours

    observed_glucose_levels = []
    insulin_rates = []
    time_steps = []
    rewards_steps = []

    cur_state = np.concatenate([[self.G0]*3, [self.ui0]*3], dtype=np.float32) # Initial state for MPC
    for t in range(len_hist):
      daytime = self.start + t * self.st / 60 # Current time of day
      time_steps.append(daytime)

      MPC = self.MPC_func(cur_state, 3, 6, self.ui0, daytime % 24) # 3, 6
      cur_state, rewards, dones, truncated, info = self.step(np.array([MPC['MPC.u']]))

      insulin_rates.append(MPC['MPC.u'])
      observed_glucose_levels.append(cur_state[2])
      rewards_steps.append(rewards)

      if dones or truncated:
        print(f"[Init Hist] Simulation ended at time step {t}.")
        break
      
    self.history= {
      'observed_glucose_levels': observed_glucose_levels,
      'insulin_rates': insulin_rates,
      'rewards_steps': rewards_steps,
      'time_steps': time_steps
    }
    # return observed_glucose_levels, insulin_rates, rewards_steps, time_steps

  def render(self, mode='console'):
    if mode != 'console':
        raise NotImplementedError()
    # Just print the state
    print(f'Step: {self.current_step}, Time of Day: {(self.start + self.current_step*self.st/60) % 24:.2f}, Glucose Level: {self.state[2]:.2f}')


# %% ActionRepeatWrapper
    
class ActionRepeatWrapper(gym.Wrapper):
  """
  A wrapper for repeating actions in an environment.
  """
  def __init__(self, env, repeat=5):
    super(ActionRepeatWrapper, self).__init__(env)
    self.repeat = repeat
    self.current_step = 0
    self.last_action = None

  def step(self, action):
    # On the first step of the cycle, record the action to repeat
    if self.current_step % self.repeat == 0:
        self.last_action = action
    # Increment the step counter
    self.current_step += 1
    # Perform the step with the last recorded action
    return self.env.step(self.last_action)

  def reset(self, **kwargs):
    # Reset the environment in the standard way
    observation = self.env.reset(**kwargs)
    # Reset the step counter and clear the last action
    self.current_step = 0
    self.last_action = None
    return observation

# %% Check the environment
if __name__ == '__main__':
  from stable_baselines3.common.env_checker import check_env

  env = DiabetesEnv()
  print(env.observation_space)
  print(env.action_space)
  obs, info = env.reset(return_info=True)
  print(obs, info)
  obs, rewards, done, truncated, info = env.step(np.array([0.5]))
  print(obs, rewards, done, truncated, info)
  print(f"obs.dtype: {obs.dtype}")
  env.render()
  env.close()
  check_env(env)
