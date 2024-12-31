from typing import Literal
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit
from gymnasium_robotics.envs.maze import AntMazeEnv as BaseAntMazeEnv
import mujoco
from mujoco.glfw import glfw


class AntMazeEnv(Env):
    def __init__(self,
                 img_size,
                 action_repeat,
                 time_limit,
                 seed,
                 maze_type=Literal['UMaze', 'Bigmaze', 'HardestMaze'],
                 maze_map=None):
        self._seed = seed
        self._action_repeat = action_repeat
        
        import gymnasium, gymnasium_robotics
        gymnasium.register_envs(gymnasium_robotics)
        env_name = f'AntMaze_{maze_type}-v5'
        kwargs = {}
        if maze_map is not None:
            kwargs['maze_map'] = maze_map
        self._base_env = gymnasium.make(env_name,
                                        render_mode='rgb_array',
                                        width=img_size,
                                        height=img_size,
                                        default_camera_config=dict(
                                            type=2,
                                            distance=160.0,
                                            lookat=[0., 0., 0.],
                                            azimuth=90,
                                            elevation=-90,
                                        ),
                                        max_episode_steps=-1,
                                        **kwargs)
        # self._base_env = BaseAntMazeEnv(render_mode='rgb_array',
        #                                 maze_map=maze_map,
        #                                 default_camera_config=dict(
        #                                     distance=16.0,
        #                                     look_at=[0., 0., 0.],
        #                                     azimuth=90,
        #                                     elevation=-90,
        #                                 ))
        self._env = TimeLimit(self._base_env, time_limit * action_repeat)
        
        # hide target point sphere
        self._base_env.unwrapped.ant_env.model.mat_rgba[self._base_env.unwrapped.target_site_id][:] = 0
        
        self.observation_space = Box(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        self.action_space = self._env.action_space
        
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        
        self._env.reset()
        self.init_qpos = self._base_env.unwrapped.data.qpos.copy()
        self.init_qvel = self._base_env.unwrapped.data.qvel.copy()
        self.goal_rendered = False
    
    def reset(self):
        self._env.reset()
        self.goal_rendered = False
        return self._env.render()
    
    def step(self, action):
        total_reward = 0
        for step in range(self._action_repeat):
            state, reward, terminated, truncated, info = self._env.step(action)
            # terminated = self.compute_success()
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
        