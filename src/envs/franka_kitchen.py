from itertools import combinations
import numpy as np
from gymnasium import Env
from gymnasium.wrappers import AddRenderObservation, TimeLimit
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import KitchenEnv


class FrankaKichenEnv(Env):
    def __init__(self,
                 img_size,
                 action_repeat,
                 time_limit):
        self._action_repeat = action_repeat
        self._base_env = KitchenEnv(render_mode='rgb_array',
                                    width=img_size,
                                    height=img_size,
                                    default_camera_config=dict(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60))
        self._env = TimeLimit(self._base_env, time_limit)
        
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        
        self.obs_element_goals, self.obs_element_indices, self.goal_configs = get_kitchen_benchmark_goals()
        self.goals = list(range(len(self.obs_element_goals)))
        self.goal_idx = 0
    
    def reset(self):
        self._env.reset()
        return self._env.render()
    
    def step(self, action):
        total_reward = 0
        for step in range(self._action_repeat):
            state, reward, truncatred, terminated, info = self._env.step(action)
            done = truncatred or terminated
            total_reward += reward
            if done:
                break
        obs = self._env.render()
        return obs, total_reward, done, info
    
    def render(self):
        return self._env.render()
    
    def close(self):
        return self._env.close()
    
    def set_goal_idx(self, idx):
        assert idx in self.goals
        self.goal_idx = idx
    
    def get_goal_obs(self):
        pass

def get_kitchen_benchmark_goals():

    object_goal_vals = {'bottom_burner' :  [-0.88, -0.01],
                        'light_switch' :  [ -0.69, -0.05],
                        'slide_cabinet':  [0.37],
                        'hinge_cabinet':   [0., 0.5],
                        'microwave'    :   [-0.5],
                        'kettle'       :   [-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]}

    object_goal_idxs = {'bottom_burner' :  [9, 10],
                        'light_switch' :  [17, 18],
                        'slide_cabinet':  [19],
                        'hinge_cabinet':  [20, 21],
                        'microwave'    :  [22],
                        'kettle'       :  [23, 24, 25, 26, 27, 28]}

    base_task_names = [ 'bottom_burner', 'light_switch', 'slide_cabinet', 'hinge_cabinet', 'microwave', 'kettle' ]

    
    goal_configs = []
    #single task
    for i in range(6):
      goal_configs.append( [base_task_names[i]])

    #two tasks
    for i,j  in combinations([1,2,3,5], 2) :
      goal_configs.append( [base_task_names[i], base_task_names[j]] )
    
    obs_element_goals = [] ; obs_element_indices = []
    for objects in goal_configs:
        _goal = np.concatenate([object_goal_vals[obj] for obj in objects])
        _goal_idxs = np.concatenate([object_goal_idxs[obj] for obj in objects])

        obs_element_goals.append(_goal)
        obs_element_indices.append(_goal_idxs)
  
    return obs_element_goals, obs_element_indices, goal_configs
