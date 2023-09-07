import os
import sys
import logging
import pathlib as pl
from copy import deepcopy
from typing import List, Dict, Optional

import cv2
import pickle
import numpy as np
from PIL import Image

from rosbag_as_dataset import RosbagReader, MissingTopicError
from ros_to_numpy_helpers import (ros_pose_to_np_se3_matrix,
                                                     ros_image_to_np,
                                                     se3_mat_to_position_and_quaterion_vec)
from ros_tf2_wrapper import (get_populated_tf2_wrapper,
                                                Tf2Wrapper)

from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from depth_image_encoding import FloatArrayToRgbImage, DEFAULT_RGB_SCALE_FACTOR
from rospy import Time
from tf2_ros import tf2

ALL_CAMERAS = [
    'left_shoulder',
    'right_shoulder',
    'front',
    'overhead',
    'wrist'
]


def check_and_mkdirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def save_img(img, path):
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, img[:, :, :3])
    elif isinstance(img, Image.Image):
        img.save(path)


def extract_list_from_rosbag_dataset(dataset:RosbagReader) -> List[Dict]:
    '''This is where the assumptions about the dataset lie.
    We are assuming that there is a series of datapoints where all topics are present.
    This function finds the start and iterates until it hits the first missing topic.
    
    Args:
        dataset: RosbagReader object
    Returns: 
        List[Dict] where each dict is a datapoint
    '''

    starting_idx = 0
    datalist = []
    for i in range(starting_idx, len(dataset)):
        try:
            datalist.append(dataset[i])
        except MissingTopicError as e:
            logging.info(f'skipped episode {i} - missing: {e.topics}.')
    return datalist


def get_dummy_observation(frame, misc):

    # pose format -> x, y, z, qx, qy, qz, qw
    dummy_pose = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0 ,1.0])

    obs = Observation(
        left_shoulder_rgb=None,
        left_shoulder_depth=None,
        left_shoulder_point_cloud=None,
        right_shoulder_rgb=None,
        right_shoulder_depth=None,
        right_shoulder_point_cloud=None,
        overhead_rgb=None,
        overhead_depth=None,
        overhead_point_cloud=None,
        wrist_rgb=None,
        wrist_depth=None,
        wrist_point_cloud=None,
        front_rgb=None,
        front_depth=None,
        front_point_cloud=None,
        left_shoulder_mask=None,
        right_shoulder_mask=None,
        overhead_mask=None,
        wrist_mask=None,
        front_mask=None,
        joint_velocities            = dummy_pose,
        joint_positions             = dummy_pose,
        joint_forces                = dummy_pose,
        gripper_open                = 0.9,
        gripper_pose                = dummy_pose,
        gripper_matrix              = dummy_pose,
        gripper_touch_forces        = None,
        gripper_joint_positions     = np.ones(2),
        task_low_dim_state          =None,
        ignore_collisions           =True, # TODO: fix
        misc=misc,
    )

    return obs


def save_keypoint(episodes:List[Dict], 
                  save_path:str, 
                  description:List[str],
                  episode_idx:int=0, 
                  depth_scale=DEFAULT_RGB_SCALE_FACTOR,
                  cameras_used=ALL_CAMERAS,
                  episode_to_rlbench_obs=get_dummy_observation
                  ):
    '''TODO: add docstring'''

     # make directories
    variation_idx = 0

    episode_path = os.path.join(save_path, 'all_variations', 'episodes', f"episode{episode_idx}")
    check_and_mkdirs(episode_path)

    base_misc = dict()

    for cam in cameras_used:
        for info in ['rgb', 'depth', 'camera_intrinsics',]:
            assert any([f'{cam}_{info}' in k for k in episodes[0].keys()]),\
                   f"Camera {cam}_{info} not found in frames!"

        image_dir = pl.Path(episode_path)/f'{cam}_rgb'
        image_dir.mkdir(parents=True, exist_ok=True)

        image_dir = pl.Path(episode_path)/f'{cam}_depth'
        image_dir.mkdir(parents=True, exist_ok=True)

        K_key = f'{cam}_camera_intrinsics'
        base_misc[K_key] = np.array(episodes[0][K_key]).reshape(3,3)
        base_misc[cam + '_camera_near'] = 0 # 0.5                           # TODO: scrutinize these values
        base_misc[cam + '_camera_far'] = 1 #4.5                            # TODO: scrutinize these values

    base_misc['keypoint_idxs'] = [len(episodes) - 1]

    observations = []
    image_keys = [f'{cam}_rgb' for cam in cameras_used]
    depth_keys = [f'{cam}_depth' for cam in cameras_used]
    for eps_idx, eps in enumerate(episodes):
        misc = deepcopy(base_misc)  
        found_image = False      
        for key, val in eps.items():
            image_dir = str(pl.Path(episode_path) / key / f'{eps_idx}.png')
            if key in image_keys: 
                save_img(cv2.cvtColor(val['image'], cv2.COLOR_RGB2BGR),
                         image_dir)
                misc[f'{key[:-4]}_camera_extrinsics'] = val['extrinsics']
                found_image = True
            elif key in depth_keys: 
                v = FloatArrayToRgbImage(val['image'], scale_factor=depth_scale)
                save_img(v, image_dir)
                found_image = True
        assert found_image, f"Could not find image in episode {eps_idx}."
        misc['keypoint'] = eps['keypoint']
        observations.append(episode_to_rlbench_obs(eps, misc))

    demo = Demo(observations, random_seed=0)
    demo.variation_number = variation_idx

    low_dim_obs_path = os.path.join(episode_path, 'low_dim_obs.pkl')
    with open(low_dim_obs_path, 'wb') as f:
        pickle.dump(demo, f)

    variation_number_path = os.path.join(episode_path, 'variation_number.pkl')
    with open(variation_number_path, 'wb') as f:
        pickle.dump(variation_idx, f)

    descriptions_path = os.path.join(episode_path, 'variation_descriptions.pkl')
    with open(descriptions_path, 'wb') as f:
        pickle.dump(description, f)

    logging.info(f"Saved {len(episodes)} frames to {save_path}")


def _setup_rosbag_dataset(data_dir, 
                          topics_and_names:dict, 
                          time_slack:Optional[float]=None, 
                          reference_topic:str=None
                          ):
    dataset = RosbagReader(data_dir, topics_and_names, 
                           permissible_asynchronisity_sec=time_slack,
                           reference_topic=reference_topic)
    def task_status_postprocessing(msg):
        return {'name': msg.task_name,
                'pose': ros_pose_to_np_se3_matrix(msg.target_pose.pose),
                'reference_frame': msg.target_pose.header.frame_id,
                'time': msg.target_pose.header.stamp,
                'status': msg.status,}
    def image_and_time(msg):
        return {'image': ros_image_to_np(msg),
                'time': msg.header.stamp}
    # instead of just getting image get its timestamp as well
    dataset.unregister_type_postprocess('sensor_msgs/Image')
    dataset.register_type_postprocess('sensor_msgs/Image', image_and_time)
    # package task status into a keypoint dict
    dataset.register_topic_postprocess('/spot/task_state', task_status_postprocessing)
    logging.info(f'Dataset initial length: {len(dataset)}')
    
    return dataset

def add_tfs_to_episodes(tf_wrapper:Tf2Wrapper, data:List[Dict], 
                    names_and_frames:Dict, ref_frame:str,
                    handle_excepts:bool=False,
                    ):
    '''Adds the extrinsics to the data. Optionally ignores missing data.
    Args:
        tf_wrapper: Tf2Wrapper object containing buffer of all tfs.
        data: List of dicts containing the data into which the extrinsics will be added.
        names_and_frames: dictionary of the data keys (or names) that are to be augmented
            with the extrinsics (or frames).
        ref_frame: The frame of reference for all the extrinsics.
    '''
    skipped = []
    for i, d in enumerate(data):
        try:
            for n, f in names_and_frames.items():
                if n not in d.keys(): 
                    # skipped.add(n)
                    time = list(d.values())[0]['time']
                    T = tf_wrapper.get(ref_frame, f, Time(secs=time.secs, nsecs=time.nsecs))
                    d[n] = T
                else:
                    time = d[n]['time']
                    T = tf_wrapper.get(ref_frame, f, Time(secs=time.secs, nsecs=time.nsecs))
                    d[n]['extrinsics'] = T
        except tf2.ExtrapolationException as e:
            if not handle_excepts:
                raise e
            else:
                logging.warning(f'In add_tfs_to_episodes: {e}')
                skipped.append(i)
        
    if len(skipped) > 0:
        logging.warning(f"Skipped {len(skipped)} frames between {skipped[0]} to {skipped[-1]} due to missing tfs.")
    return skipped


def keypoints_from_frame(data:List[Dict], tf_wrapper:Tf2Wrapper, ref_frame:str):
    '''Transforms all keypoint poses to the reference frame.'''
    for d in data:
        k = d['keypoint']
        #### NOTE: TEMPORARY FIX!!!
        ## This should be the data time.
        time = Time(secs=d['left_shoulder_rgb']['time'].secs, nsecs=d['left_shoulder_rgb']['time'].nsecs)
        # time = Time(secs=k['time'].secs, nsecs=k['time'].nsecs)
        T = tf_wrapper.get(ref_frame, k['reference_frame'], time)
        k['pose'] = se3_mat_to_position_and_quaterion_vec(T @ k['pose'])
        k['reference_frame'] = ref_frame










def rosbag_to_rlbench(data_dir:pl.Path, 
                      out_dir:pl.Path, 
                      topics_and_names:Dict, 
                      names_and_frames:Dict,
                      reference_frame:str, 
                      task_description:List[str]=["no lang goal defined"],
                      episode_idx:int = 0,
                      depth_scale = ( ( 2**22 ) - 1.0 ),
                      cameras_used = ALL_CAMERAS,
                      time_slack:Optional[float]=None,
                      reference_topic:Optional[str]=None,
                      reference_frame_T:np.ndarray=np.eye(4),
                      ) -> None:
    '''
    Args:
        names_and_frames: dictionary of all the datapoints' names 
            and tf frame names. This list indicates which datapoints
            we care to associate with a pose. Mainly intended for
            camera data extrinsics.
        reference: The frame of reference for all the extrinsics.
    '''
    logging.info('.'*3 + f'processing: {data_dir}' + '.'*3)
    tf_data = get_populated_tf2_wrapper(data_dir)
    tf_data.register_getter_hook(lambda x: reference_frame_T @ x, reference_frame)
    dataset = _setup_rosbag_dataset(data_dir, topics_and_names,
                                    time_slack=time_slack,
                                    reference_topic=reference_topic)

    logging.info(f'depth scale: {depth_scale}')
    episodes = extract_list_from_rosbag_dataset(dataset)
    add_tfs_to_episodes(tf_data, 
                        episodes, 
                        names_and_frames, 
                        reference_frame)
    keypoints_from_frame(episodes, 
                         tf_data, 
                         reference_frame)
    save_keypoint(episodes, 
                  out_dir, 
                  task_description, 
                  episode_idx, 
                  depth_scale=depth_scale,
                  cameras_used=cameras_used)
    logging.info('.'*5 + 'Done!' + '.'*5)


def rosbags_to_rlbench_episodes(data_dir:pl.Path, 
                                out_dir:pl.Path, 
                                topics_and_names:Dict,
                                names_and_frames, 
                                reference_frame,
                                task_description:List[str]=["no lang goal defined"],
                                depth_scale = ( ( 2**22 ) - 1.0 ),
                                cameras_used:List[str]=ALL_CAMERAS,
                                time_slack:Optional[float]=None,
                                reference_topic:Optional[str]=None,
                                reference_frame_T:np.ndarray=np.eye(4),
                                ) -> None:

    for i, data_path in enumerate(data_dir.glob('*.bag')):
        rosbag_to_rlbench(data_path, 
                          out_dir/data_dir.name, 
                          topics_and_names, 
                          names_and_frames, 
                          reference_frame,
                          task_description,
                          i,
                          depth_scale=depth_scale,
                          cameras_used=cameras_used,
                          time_slack=time_slack,
                          reference_topic=reference_topic,
                          reference_frame_T=reference_frame_T,
                          )
    logging.getLogger(__name__).info('Done!')
