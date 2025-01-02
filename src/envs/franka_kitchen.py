from itertools import combinations
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.wrappers import AddRenderObservation, TimeLimit
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import KitchenEnv


class FrankaKichenEnv(Env):
    def __init__(self,
                 img_size,
                 action_repeat,
                 time_limit,
                 seed):
        self._seed = seed
        self._action_repeat = action_repeat
        self._base_env = KitchenEnv(render_mode='rgb_array',
                                    width=img_size,
                                    height=img_size,
                                    default_camera_config=dict(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60))
        self._env = TimeLimit(self._base_env, time_limit * action_repeat)
        
        self.observation_space = Box(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        self.action_space = self._env.action_space
        
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        
        self.obs_element_goals, self.obs_element_indices, self.goal_configs = get_kitchen_benchmark_goals()
        self.goals = list(range(len(self.obs_element_goals)))
        self.goal_idx = -1
        
        self._env.reset(seed=self._seed)
        self.init_qpos = self._base_env.data.qpos.copy()
        self.init_qvel = self._base_env.data.qvel.copy()
        self.goal_rendered = False
    
    def reset(self):
        self._env.reset(seed=self._seed)
        self.goal_rendered = False
        return self._env.render()
    
    def step(self, action):
        total_reward = 0
        for step in range(self._action_repeat):
            state, reward, terminated, truncated, info = self._env.step(action)
            terminated = self.compute_success()
            done = truncated or terminated
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
        assert (idx in self.goals) or (idx == -1)
        self.goal_idx = idx
        self.goal_rendered = False
    
    def get_goal_obs(self):
        if self.goal_idx == -1:
            return None
        
        if self.goal_rendered:
            return self.rendered_goal_obs
        
        element_indices = self.obs_element_indices[self.goal_idx]
        element_values = self.obs_element_goals[self.goal_idx]
        
        backup_qpos = self._base_env.data.qpos.copy()
        backup_qvel = self._base_env.data.qvel.copy()
        
        qpos = self.init_qpos.copy()
        qpos[element_indices] = element_values
        self._base_env.robot_env.set_state(qpos, np.zeros_like(self.init_qvel))
        
        goal_obs = self._env.render()
        
        self._base_env.robot_env.set_state(backup_qpos, backup_qvel)
        
        self.goal_rendered = True
        self.rendered_goal_obs = goal_obs
        return goal_obs
    
    def compute_success(self):
        if self.goal_idx == -1:
            return False
        
        qpos = self._base_env.data.qpos.copy()
        
        element_indices = self.obs_element_indices[self.goal_idx]
        element_values = self.obs_element_goals[self.goal_idx]
        goal_qpos = self.init_qpos.copy()
        goal_qpos[element_indices] = element_values
        
        per_obj_success = {
            'bottom_burner' : ((qpos[9]<-0.38) and (goal_qpos[9]<-0.38)) or ((qpos[9]>-0.38) and (goal_qpos[9]>-0.38)),
            'top_burner':    ((qpos[13]<-0.38) and (goal_qpos[13]<-0.38)) or ((qpos[13]>-0.38) and (goal_qpos[13]>-0.38)),
            'light_switch':  ((qpos[17]<-0.25) and (goal_qpos[17]<-0.25)) or ((qpos[17]>-0.25) and (goal_qpos[17]>-0.25)),
            'slide_cabinet' :  abs(qpos[19] - goal_qpos[19])<0.1,
            'hinge_cabinet' :  abs(qpos[21] - goal_qpos[21])<0.2,
            'microwave' :      abs(qpos[22] - goal_qpos[22])<0.2,
            'kettle' : np.linalg.norm(qpos[23:25] - goal_qpos[23:25]) < 0.2
        }
        task_objects = self.goal_configs[self.goal_idx]
        
        success = 1
        for _obj in task_objects:
            success *= per_obj_success[_obj]
        
        return bool(success)

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
                        'kettle'       :  [23, 24, 25, 26, 27, 28, 29]}

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
