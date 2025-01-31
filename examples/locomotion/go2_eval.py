import argparse
import os
import pickle

import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.utils.geom import transform_by_quat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("-v", "--view", type=str, default="side", help="Camera view [side/front]")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    
    # 在创建环境前修改配置
    env_cfg["visualize_camera"] = True
    env_cfg["max_visualize_FPS"] = 60
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    with torch.no_grad():
        env.cam.start_recording()  # 开始录制
        for _ in range(300):  # 录制300帧（约6秒）
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            
            # 设置摄像头视角
            if args.view == "side":
                env.cam.set_pose(
                    pos=(2.0, 3.0, 1.0),
                    lookat=(0.3, 0, 0.3),
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
            
            env.cam.render()
        
        # 生成带实验名称的文件名
        filename = f"video_{args.view}_{args.exp_name.replace('-', '_')}.mp4"
        env.cam.stop_recording(save_to_filename=filename, fps=env.env_cfg["max_visualize_FPS"])


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
