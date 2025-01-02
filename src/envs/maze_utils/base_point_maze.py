import xml.etree.ElementTree as ET
from os import path
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import mujoco
from mujoco.glfw import glfw
from gymnasium import spaces
from gymnasium_robotics.envs.maze.point import PointEnv
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames

from .maze import MazeEnv


class BasePointMazeEnv(MazeEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        maze_map: List[List[Union[str, int]]] = None,
        maze_color_map: List[List[Union[str]]] = None,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        reset_target: bool = False,
        xml_file: Union[str, None] = None,
        max_episode_steps: int = 1000,
        width: int = 128,
        height: int = 128,
        **kwargs,
    ):
        self.width = width
        self.height = height
        
        if xml_file is None:
            ant_xml_file_path = path.join(
                path.dirname(sys.modules[PointEnv.__module__].__file__), "../assets/point/point.xml"
            )
        else:
            ant_xml_file_path = xml_file

        super().__init__(
            agent_xml_path=ant_xml_file_path,
            maze_map=maze_map,
            maze_color_map=maze_color_map,
            maze_size_scaling=1,
            maze_height=0.5,
            reward_type=reward_type,
            continuing_task=continuing_task,
            reset_target=reset_target,
            **kwargs,
        )
        self.point_env = PointEnv(
            xml_file=self.tmp_xml_file_path,
            render_mode="rgb_array",
            **kwargs,
        )

        # ゴール位置のサイト ID を取得
        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space

        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                achieved_goal=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                desired_goal=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            )
        )

        self.render_mode = render_mode
        EzPickle.__init__(
            self,
            render_mode,
            maze_map,
            reward_type,
            continuing_task,
            reset_target,
            max_episode_steps,
            **kwargs,
        )
        
        # GLFW の初期化
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # GLFW ウィンドウの作成（オフスクリーン用）
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # ウィンドウを非表示にする
        window = glfw.create_window(width, height, "MuJoCo Offscreen", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        # OpenGL コンテキストを現在のスレッドに設定
        glfw.make_context_current(window)

        # MuJoCo 描画コンテキストの作成
        self.context = mujoco.MjrContext(self.point_env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # カメラ設定
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        # シーン設定
        self.scene = mujoco.MjvScene(self.point_env.model, maxgeom=1000)


    def reset(self, *, seed: Optional[int] = None, **kwargs):
        """
        環境をリセットし、初期状態を返す
        """
        # MazeEnv のリセットを呼び出して初期化
        super().reset(seed=seed, **kwargs)

        # ゴールと初期位置を反映
        self.point_env.init_qpos[:2] = self.reset_pos
        self.point_env.reset(seed=seed)

        self.update_target_site_pos()

        return self._get_obs()

    def step(self, action):
        """
        1ステップ実行
        """
        _, _, _, _, info = self.point_env.step(action)
        obs = self._get_obs()
        achieved_goal = self.point_env.data.qpos[:2]
        reward = self.compute_reward(achieved_goal, self.goal, {})
        terminated = self.compute_terminated(achieved_goal, self.goal, {})
        truncated = False
        info = {"success": terminated}
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        現在の観測を取得
        """

        width, height = self.width, self.height
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
        depth_array = np.zeros((height, width), dtype=np.float32)

        
        self.camera.lookat[:2] = self.point_env.data.qpos[:2]
        self.camera.lookat[2] = 0
        self.camera.distance = 4.0  # 必要に応じて調整
        self.camera.azimuth = 90
        self.camera.elevation = -90

        mujoco.mjv_updateScene(
            self.point_env.model,
            self.point_env.data,
            mujoco.MjvOption(),
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

        # 描画
        mujoco.mjr_render(
            mujoco.MjrRect(0, 0, width, height),
            self.scene,
            self.context,
        )

        # ピクセルデータを取得
        mujoco.mjr_readPixels(rgb_array, depth_array, mujoco.MjrRect(0, 0, width, height), self.context)
        rgb_array = np.flipud(rgb_array)

        # ゴールと観測データを取得
        achieved_goal = self.point_env.data.qpos[:2]

        # return {
        #     "observation": rgb_array,
        #     "achieved_goal": achieved_goal,
        #     "desired_goal": self.goal,
        # }
        return rgb_array


    def update_target_site_pos(self):
        """
        ゴール位置を MuJoCo シミュレーションに反映
        """
        pos = self.goal  # ゴール位置を設定
        self.point_env.model.site_pos[self.target_site_id] = np.append(
            pos, self.maze.maze_height / 2 * self.maze.maze_size_scaling
        )

    def render(self):
        """
        人間向けレンダリング
        """
        return self.point_env.render(mode="human")

    def close(self):
        super().close()
        self.point_env.close()
        # GLFW を終了
        glfw.terminate()