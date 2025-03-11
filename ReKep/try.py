# from openai import OpenAI

# client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")  # vLLM 不需要 API Key

# response = client.completions.create(
#     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#     prompt="""Traceback (most recent call last):
#   File "/home/lenovo/project/ReKep/main.py", line 379, in <module>
#     main = Main(scene_file, visualize=args.visualize)
#   File "/home/lenovo/project/ReKep/main.py", line 40, in __init__
#     self.env = ReKepOGEnv(global_config['env'], scene_file, verbose=False)
#   File "/home/lenovo/project/ReKep/environment.py", line 69, in __init__
#     self.og_env = og.Environment(dict(scene=self.config['scene'], robots=[self.config['robot']['robot_config']], env=self.config['og_sim']))
#   File "/home/lenovo/project/OmniGibson/omnigibson/utils/python_utils.py", line 93, in wrapper
#     func(*values.args, **values.kwargs)
#   File "/home/lenovo/project/OmniGibson/omnigibson/envs/env_base.py", line 103, in __init__
#     og.launch(
#   File "/home/lenovo/project/OmniGibson/omnigibson/simulator.py", line 1780, in _launch_simulator
#     Simulator(*args, **kwargs)
#   File "/home/lenovo/project/OmniGibson/omnigibson/simulator.py", line 295, in __init__
#     self._validate_dts(physics_dt, rendering_dt, sim_step_dt)
#   File "/home/lenovo/project/OmniGibson/omnigibson/simulator.py", line 508, in _validate_dts
#     assert math.isclose(
# AssertionError: Rendering dt (0.03333333333333333) must be a multiple of physics dt (0.01)
# """,
#     max_tokens=10000,
#     temperature=0.7
# )
# print(response.choices[0].text)

import sys
import time
sys.path.append("~/project/ReKep/")
from environment import ReKepOGEnv
import omnigibson as og
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

global_config = get_config(config_path="./configs/config.yaml")
env = ReKepOGEnv(global_config['env'],scene_file='./configs/og_scene_file_drawer.json',verbose=False)
env.reset()

# 持续显示环境
while True:
    # 在每个时间步进行仿真
    
    # 渲染并显示
    og.sim.render()  # 这是关键：持续渲染
    
    # 如果环境结束（比如任务完成或达到最大步数），重

    time.sleep(0.01)  # 控制循环速度，可以根据需要调整

# import omnigibson as og
# og.sim.initialize()
# og.sim.viewer()