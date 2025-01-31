import argparse
import os
import pickle

import torch
from go2_env import Go2Env

import genesis as gs
import numpy as np
from genesis.utils.geom import transform_by_quat


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 70.0,
        "kd": 3.0,
        # termination
        "termination_if_roll_greater_than": 1000,  # degree
        "termination_if_pitch_greater_than": 1000,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.35],
        "base_init_quat": [0.0, 0.0, 0.0, 1.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.5,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 60,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "reward_scales": {},
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0, 0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


class BackflipEnv(Go2Env):
    def get_observations(self):
        phase = torch.pi * self.episode_length_buf[:, None] / self.max_episode_length
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                self.last_actions,  # 12
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2),
                torch.cos(phase / 2),
                torch.sin(phase / 4),
                torch.cos(phase / 4),
            ],
            axis=-1,
        )

        return self.obs_buf

    def step(self, actions):
        super().step(actions)
        self.get_observations()
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="single")
    parser.add_argument("-v", "--view", type=str, default="side")
    args = parser.parse_args()

    gs.init()


    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    if args.exp_name == "single":
        env_cfg["episode_length_s"] = 2
    elif args.exp_name == "double":
        env_cfg["episode_length_s"] = 3
    else:
        raise RuntimeError

     # visualize the target
    env_cfg["visualize_target"] = True
    # for video recording
    env_cfg["visualize_camera"] = True
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60

    env = BackflipEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    policy = torch.jit.load(f"./backflip/{args.exp_name}.pt")
    policy.to(device="cuda:0")

    obs, _ = env.reset()
    with torch.no_grad():
        env.cam.start_recording()
        for i in range(100):
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            
            if args.view == "side":
                # 侧视角：从右前方45度观察全身
                env.cam.set_pose(
                    pos=(2.0, 3.0, 1.0),  # X向右，Y向前，Z向上
                    lookat=(0.3, 0, 0.3),    # 观察基座中心稍高位置
                )
            elif args.view == "front":
                base_pos = env.base_pos[0].cpu().numpy()
                base_quat = env.base_quat[0].cpu().numpy()
                
                # 摄像头位置：基座前方0.3米，高度0.15米
                head_offset = transform_by_quat(
                    torch.tensor([[-0.3, 0.0, 0.15]], device=env.device),
                    torch.tensor([base_quat], device=env.device)
                )[0].cpu().numpy()
                cam_pos = base_pos + head_offset
                
                # 观察点：基座前方1米，地面下方0.2米
                lookat_offset = transform_by_quat(
                    torch.tensor([[1.0, 0.0, -0.2]], device=env.device),
                    torch.tensor([base_quat], device=env.device)
                )[0].cpu().numpy()
                lookat_pos = base_pos + lookat_offset
                
                # 先设置姿态
                env.cam.set_pose(
                    pos=cam_pos,
                    lookat=lookat_pos,
                )
                # 再单独设置FOV参数
                env.cam.set_params(fov=80)  # 设置80度广角
            
            # 统一渲染（两个视角都需要）
            env.cam.render()
        filename = f"video_{args.view}.mp4"
        env.cam.stop_recording(save_to_filename=filename, fps=env_cfg["max_visualize_FPS"])
if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_backflip.py -e single
python examples/locomotion/go2_backflip.py -e double
"""
