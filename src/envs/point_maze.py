import numpy as np
from gymnasium import Env
from gymnasium import spaces
from gymnasium_robotics.envs.maze.maps import *
from gymnasium.wrappers import TimeLimit
from pyvirtualdisplay import Display

from .maze_utils.base_point_maze import BasePointMazeEnv
from .maze_utils.maps import *

class PointMazeEnv(Env):
    def __init__(self,
                 img_size,
                 action_repeat,
                 time_limit,
                 seed,
                 maze_map = U_MAZE,
                 maze_color_map = U_MAZE_COLOR):
        display = Display(visible=0, size=(1024, 768))
        display.start()
        
        self._seed = seed
        self._action_repeat = action_repeat
        self._base_env = BasePointMazeEnv(
            render_mode='rgb_array',
            width=img_size,
            height=img_size,
            maze_map=maze_map,
            maze_color_map=maze_color_map,
            max_episode_steps=-1,
        )
        self._env = TimeLimit(self._base_env, time_limit * action_repeat)
        
        self.observation_space = spaces.Box(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        self.action_space = self._env.action_space
        
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        
        self.goal_locations = self.get_benchmark_goals()
        self.goals = list(range(len(self.goal_locations)))
        self.goal_idx = -1
        
        self._env.reset(seed=self._seed)
        self.init_qpos = self._base_env.point_env.data.qpos.copy()
        self.init_qvel = self._base_env.point_env.data.qvel.copy()
        self.goal_rendered = False
        self.reset_pos = None
    
    def get_benchmark_goals(self):
        goal_locations = self._base_env.maze.unique_goal_locations
        idxs = np.random.choice(np.arange(len(goal_locations)), size=min(20, len(goal_locations)), replace=False)
        goals = [self._base_env.maze.unique_goal_locations[idx] for idx in idxs]
        return goals
    
    def reset(self):
        options = {}
        if self.reset_pos is not None:
            options['reset_cell'] = self.reset_pos
        self._env.reset(seed=self._seed, options=options)
        self.goal_rendered = False
        return self._base_env._get_obs()
    
    def step(self, action):
        total_reward = 0
        for step in range(self._action_repeat):
            state, reward, terminated, truncated, info = self._env.step(action)
            terminated = self.compute_success()
            done = truncated or terminated
            total_reward += reward
            if done:
                break
        obs = self._base_env._get_obs()
        return obs, total_reward, done, info
    
    def render(self):
        return self._base_env._get_obs()
    
    def close(self):
        return self._env.close()
    
    def set_goal_idx(self, idx):
        assert (idx in self.goals) or (idx == -1)
        self.goal_idx = idx
        self.goal_rendered = False
        if idx == -1:
            self.reset_pos = None
        else:
            self.reset_pos = [1, 1]
    
    def get_goal_obs(self):
        if self.goal_idx == -1:
            return None
        
        if self.goal_rendered:
            return self.rendered_goal_obs
        
        goal_location = self.goal_locations[self.goal_idx]
        
        backup_qpos = self._base_env.point_env.data.qpos.copy()
        backup_qvel = self._base_env.point_env.data.qvel.copy()
        
        qpos = self.init_qpos.copy()
        qpos[:2] = goal_location
        self._base_env.point_env.set_state(qpos, np.zeros_like(self.init_qvel))
        
        goal_obs = self._base_env._get_obs()
        
        self._base_env.point_env.set_state(backup_qpos, backup_qvel)
        
        self.goal_rendered = True
        self.rendered_goal_obs = goal_obs
        return goal_obs
    
    def compute_success(self):
        if self.goal_idx == -1:
            return False
        
        qpos = self._base_env.point_env.data.qpos.copy()
        achieved_goal = qpos[:2]
        desired_goal = self.goal_locations[self.goal_idx]
        return bool(np.linalg.norm(achieved_goal - desired_goal) <= 0.45)
