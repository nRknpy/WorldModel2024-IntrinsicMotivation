def env_factory(name, seed, img_size, action_repeat, time_limit, **kwargs):
    match name:
        case 'FrankaKitchen':
            from .franka_kitchen import FrankaKichenEnv
            env = FrankaKichenEnv(
                img_size,
                action_repeat,
                time_limit,
                seed,
                **kwargs,
            )
        case 'AntMaze':
            from .ant_maze import AntMazeEnv, LARGE_MAZE, LARGE_MAZE_COLOR
            env = AntMazeEnv(
                img_size,
                action_repeat,
                time_limit,
                seed,
                LARGE_MAZE,
                LARGE_MAZE_COLOR,
                **kwargs,
            )
        case 'PointMaze':
            from .point_maze import PointMazeEnv, LARGE_MAZE, LARGE_MAZE_COLOR
            env = PointMazeEnv(
                img_size,
                action_repeat,
                time_limit,
                seed,
                LARGE_MAZE,
                LARGE_MAZE_COLOR,
                **kwargs,
            )
    return env
