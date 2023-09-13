
import yaml
import logging
import pathlib as pl
import sys
from ros_tf2_wrapper import get_populated_tf2_wrapper
from rosbag_to_rlbench import (_setup_rosbag_dataset, 
                               extract_list_from_rosbag_dataset, 
                               add_tfs_to_episodes, 
                               keypoints_from_frame,
                               save_keypoint)
from rlbench.backend.observation import Observation
import numpy as np

class Config:
    def __init__(self, cfg: dict = {}):
        self.__dict__.update(cfg)

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONFIG_DIR = './configs/peract.yaml'



# TODO: What about ignore collision and low_dim_state and time?????

def get_rlbench_obs(eps, misc):
     # pose format -> x, y, z, qx, qy, qz, qw
    dummy_pose = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0 ,1.0])

    gPO_ratio = eps['gripper_input']['gPO'] / 255.0
    obs = Observation(
        left_shoulder_rgb=None,          left_shoulder_depth=None,
        left_shoulder_point_cloud=None,  right_shoulder_rgb=None,
        right_shoulder_depth=None,       right_shoulder_point_cloud=None,
        overhead_rgb=None,               overhead_depth=None,
        overhead_point_cloud=None,       wrist_rgb=None,
        wrist_depth=None,                wrist_point_cloud=None,
        front_rgb=None,                  front_depth=None,
        front_point_cloud=None,          left_shoulder_mask=None,
        right_shoulder_mask=None,        overhead_mask=None,
        wrist_mask=None,                 front_mask=None,
        joint_velocities            = eps['joint_states']['velocities'],
        joint_positions             = eps['joint_states']['positions'],
        joint_forces                = eps['joint_states']['efforts'],
        gripper_open                = gPO_ratio > 0.9,
        gripper_pose                = dummy_pose,
        gripper_matrix              = eps['end_effector'],
        gripper_touch_forces        = None,
        gripper_joint_positions     = [gPO_ratio/2, gPO_ratio/2],
        task_low_dim_state          = None,
        ignore_collisions           = True, # TODO: fix
        misc=misc,)
    return obs


# Load yaml config
assert pl.Path(CONFIG_DIR).exists(), f'Config file not found: {CONFIG_DIR}'
cfg = Config(yaml.safe_load(open(CONFIG_DIR, 'r')))

data_dir = pl.Path(cfg.data_dir)
out_dir = pl.Path(cfg.save_path)

# For every task folder and its description
for task_folder, description in zip(cfg.task_folders, cfg.task_descriptions):
    task_dir = data_dir/task_folder
    logging.info('-'*30)
    logging.info('.'*5 + f'Processing task: {task_folder}' + '.'*5)
    # For every rosbag/demo in folder
    for demo_idx, demo_path in enumerate(task_dir.glob('*.bag')):
        logging.info('\n'+'.'*3 + f'processing: {demo_path}' + '.'*3)
        logging.info(f'depth scale: {cfg.depth_scale}')
        
        tf_data = get_populated_tf2_wrapper(demo_path, slack_sec=cfg.topic_time_slack)
        tf_data.register_getter_hook(lambda x: cfg.reference_frame_T @ x, cfg.reference_frame)
        dataset = _setup_rosbag_dataset(demo_path, cfg.topics_and_names,
                                        time_slack=cfg.topic_time_slack,
                                        reference_topic=cfg.reference_topic)
        dataset.register_topic_postprocess('/right/Robotiq2FGripperRobotInput', 
                                           lambda a: {'gACT': a.gACT, 'gGTO': a.gGTO, 'gSTA': a.gSTA,
                                                      'gOBJ': a.gOBJ, 'gFLT': a.gFLT,'gPR': a.gPR,
                                                      'gPO': a.gPO,'gCU': a.gCU,} 
                                           )
        episodes = extract_list_from_rosbag_dataset(dataset)
        
        skipped = add_tfs_to_episodes(tf_data, episodes, cfg.names_and_frames, cfg.reference_frame,)
        episodes = [e for i, e in enumerate(episodes) if i not in skipped]  # Remove points with no tfs
        logging.info(f'Saving {len(episodes)} episodes')
        # keypoints_from_frame(episodes, tf_data, cfg.reference_frame)
        
        save_keypoint(episodes, 
                      out_dir/pl.Path(task_folder).name, 
                      description, 
                      demo_idx, 
                      depth_scale=cfg.depth_scale,
                      cameras_used=cfg.cameras_used,
                      episode_to_rlbench_obs=get_rlbench_obs,
                      near_far_planes=cfg.near_far_planes,)
        logging.info('.'*3 + f'Completed demo {demo_idx}' + '.'*3)
    logging.info('Done with task ')

