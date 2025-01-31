import genesis as gs
import numpy as np
from genesis.utils.geom import transform_by_quat
import torch

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=False,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

########################## 添加双摄像头 ##########################
# 侧视摄像头（固定视角）
cam_side = scene.add_camera(
    res=(1280, 720),
    pos=(1.5, -1.5, 1.0),
    lookat=(0.65, 0.0, 0.2),
    fov=60,
)

# 夹爪摄像头（跟随视角）
cam_gripper = scene.add_camera(
    res=(1280, 720),
    pos=(0.65, 0.0, 0.25),
    lookat=(0.65, 0.0, 0.1),
    fov=80,
)

########################## build ##########################
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# set control gains
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

end_effector = franka.get_link("hand")

# move to pre-grasp pose
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)
# gripper open pos
qpos[-2:] = 0.04
path = franka.plan_path(
    qpos_goal=qpos,
    num_waypoints=200,  # 2s duration
)

########################## 录制视频逻辑 ##########################
# 开始录制
cam_side.start_recording()
cam_gripper.start_recording()

# execute the planned path
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()
    
    # 更新夹爪摄像头视角
    current_pos = franka.get_link("hand").get_pos().cpu().numpy()
    current_quat = franka.get_link("hand").get_quat().cpu().numpy()
    
    # 计算摄像头位置（夹爪后方0.1米，上方0.05米）
    offset = np.array([-0.1, 0.0, 0.1])  # X轴负方向为夹爪后方
    gripper_cam_pos = current_pos + transform_by_quat(offset, current_quat)
    
    # 计算观察点（夹爪前方0.3米）
    look_offset = np.array([0.3, 0.0, 0.0])  # X轴正方向为夹爪前方
    gripper_lookat = current_pos + transform_by_quat(look_offset, current_quat)
    
    cam_gripper.set_pose(
        pos=gripper_cam_pos,
        lookat=gripper_lookat
    )
    cam_gripper.set_params(fov=70)  # 微调视野范围
    
    # 渲染双视角
    cam_side.render()
    cam_gripper.render()

# allow robot to reach the last waypoint
for i in range(100):
    scene.step()

# reach
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.130]),
    quat=np.array([0, 1, 0, 0]),
)
print(qpos)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(100):
    scene.step()

# grasp
franka.control_dofs_position(qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)

for i in range(100):
    scene.step()

# lift
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]),
    quat=np.array([0, 1, 0, 0]),
)
print(qpos)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(200):
    scene.step()

# 停止录制并保存
cam_side.stop_recording(save_to_filename="grasp_side.mp4", fps=60)
cam_gripper.stop_recording(save_to_filename="grasp_gripper.mp4", fps=60)
