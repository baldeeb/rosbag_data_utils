import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data_preprocessing.ros_tf2_wrapper import (get_populated_tf2_wrapper)
from data_preprocessing.rosbag_to_rlbench import (keypoints_from_frame, 
                                                  extract_list_from_rosbag_dataset, 
                                                  add_tfs_to_episodes, 
                                                  _setup_rosbag_dataset)

from quaternion import quaternion, as_rotation_matrix, from_rotation_matrix

###### Settings ##########################################
SHOW_2D = True
SHOW_3D = False
REFERENCE_FRAME = 'body' # 'map' or 'body'
PERSPECTIVE = 'other'  # 'waypoint' or other
                            # FOR VISUALIZATION - using map frame will yield static waypoints and meaningless visuals.
NUMBER_POINTS = False
##########################################################

if __name__ == '__main__':


    logging.basicConfig(level=logging.INFO)

    # data_dir = pl.Path('/media/baldeeb/ssd2/Data/SpotBags/2023_08_01')
    # topics_and_names = {
    #     '/spot/camera/frontleft/image':                         'left_shoulder_rgb',
    #     '/spot/depth/frontleft/depth_in_visual':                'left_shoulder_depth',
    #     '/spot/camera/frontleft/camera_info':                   'left_shoulder_camera_intrinsics',
    #     # '/spot/depth/frontleft/depth_in_visual/camera_info':    '',

    #     '/spot/camera/frontright/image':                        'right_shoulder_rgb',
    #     '/spot/depth/frontright/depth_in_visual':               'right_shoulder_depth',
    #     '/spot/camera/frontright/camera_info':                  'right_shoulder_camera_intrinsics',
    #     # '/spot/depth/frontright/depth_in_visual/camera_info':   '',

    #     '/spot/camera/right/image':                             'front_rgb' ,
    #     '/spot/depth/right/image':                              'front_depth',
    #     '/spot/camera/right/camera_info':                       'front_camera_intrinsics',

    #     '/spot/camera/left/image':                              'overhead_rgb' ,
    #     '/spot/depth/left/image':                               'overhead_depth',
    #     '/spot/camera/left/camera_info':                        'overhead_camera_intrinsics',

    #     # '/spot/camera/hand_color/image':                        'wrist_rgb' ,
    #     # '/spot/depth/hand/depth_in_color':                      'wrist_depth',
    #     # '/spot/camera/hand_color/camera_info':                  'wrist_camera_intrinsics',

    #     '/spot/task_state':                                     'keypoint',
    #     }



    # data_dir = pl.Path('/media/baldeeb/ssd2/Data/SpotBags/2023_08_08')

    # topics_and_names = {
    #     '/spot/camera/frontleft/image':                         'left_shoulder_rgb',
    #     '/spot/depth/frontleft_in_visual/image':                'left_shoulder_depth',
    #     '/spot/camera/frontleft/camera_info':                   'left_shoulder_camera_intrinsics',

    #     '/spot/camera/frontright/image':                        'right_shoulder_rgb',
    #     '/spot/depth/frontright_in_visual/image':               'right_shoulder_depth',
    #     '/spot/camera/frontright/camera_info':                  'right_shoulder_camera_intrinsics',

    #     '/spot/camera/right/image':                             'front_rgb' ,
    #     '/spot/depth/right/image':                              'front_depth',
    #     '/spot/camera/right/camera_info':                       'front_camera_intrinsics',

    #     '/spot/camera/left/image':                              'overhead_rgb' ,
    #     '/spot/depth/left/image':                               'overhead_depth',
    #     '/spot/camera/left/camera_info':                        'overhead_camera_intrinsics',

    #     # '/spot/camera/hand_color/image':                        'wrist_rgb' ,
    #     # '/spot/depth/hand_in_visual/image':                      'wrist_depth',
    #     # '/spot/camera/hand_color/camera_info':                  'wrist_camera_intrinsics',

    #     '/spot/task_state':                                     'keypoint',
    #     }

    # reference_frame = REFERENCE_FRAME
    # names_and_frames = {
    #     'left_shoulder_rgb':    'frontleft_fisheye',
    #     'right_shoulder_rgb':   'frontright_fisheye',
    #     'front_rgb':            'right_fisheye',
    #     'overhead_rgb':         'left_fisheye',
    #     # 'wrist_rgb':            'hand_color_image_sensor',
    #     'body_T_vision':        'vision',  # The waypoint is in the vision frame
    #     }
    


    data_dir = pl.Path('/media/baldeeb/ssd2/Data/SpotBags/2023_08_25_eval')

    topics_and_names = {
        '/camera1/color/image_raw':                              'left_shoulder_rgb',
        # '/camera1/depth/image_rect_raw':                         'left_shoulder_depth',
        '/camera1/aligned_depth_to_color/image_raw':             'left_shoulder_depth',
        '/camera1/color/camera_info':                            'left_shoulder_camera_intrinsics',

        '/camera2/color/image_raw':                              'right_shoulder_rgb',
        # '/camera2/depth/image_rect_raw':                         'right_shoulder_depth',
        '/camera2/aligned_depth_to_color/image_raw':             'right_shoulder_depth',
        '/camera2/color/camera_info':                            'right_shoulder_camera_intrinsics',

        '/spot/task_state':                                     'keypoint',
    }
    time_slack = 0.1
    reference_frame = REFERENCE_FRAME
    names_and_frames = {
        'left_shoulder_rgb':    'camera1_color_optical_frame',
        'right_shoulder_rgb':   'camera2_color_optical_frame',
        }


    def to_Rt(qt, pt):
        Rt = np.eye(4)
        Rt[:3, :3] = as_rotation_matrix(qt)
        Rt[:3, 3] = pt
        Rt_inv = np.linalg.inv(Rt)
        return Rt, Rt_inv

    def get_pose_details(pose,
                         perspective='body'  # 'body' or 'waypoint'
                         ):
        x,y,z,qx,qy,qz,qw = pose
        offset = np.array([0.05, 0.0, 0.0])
        q = quaternion(qw, qx, qy, qz)
        p = np.array([x,y,z])

        waypointTperspective, perspectiveTwaypoint = to_Rt(q, p)
        
        def _pt_and_dir(T):
            return T[:3, 3], T[:3, :3] @ offset

        if perspective == 'waypoint':
            ## To display the body in the waypoint frame
            p, dir_pt =  _pt_and_dir(perspectiveTwaypoint)
        else:
            ## To display the waypoint in the body frame
            p, dir_pt =  _pt_and_dir(waypointTperspective)

        # bodyTwaypoint = as_rotation_matrix(q)
        # dir_pt = bodyTwaypoint @ offset
        return p, np.array(dir_pt)

    group = []
    bag_paths = list(data_dir.glob('*.bag'))
    for i, data_path in enumerate(bag_paths):
        logging.info('*'*20 + f'\nprocessing: {data_path}')
        # Accumulates all tfs from a bag in one buffer
        tf_data = get_populated_tf2_wrapper(data_path, slack_sec=0.1)

        # Extracts the data from the bag as a list of dicts.
        #   The dict is described by topics_and_names.
        #   The result discards any leading or training frames that had missing data.
        dataset = _setup_rosbag_dataset(data_path, topics_and_names, time_slack=time_slack, reference_topic='/spot/task_state')
        episode = extract_list_from_rosbag_dataset(dataset)
        skipped = add_tfs_to_episodes(tf_data, episode, names_and_frames, reference_frame, handle_excepts=True)

        episode = [e for i, e in enumerate(episode) if i not in skipped]
        if len(episode) < 2:
            logging.warning(f"Skipping {data_path} due to insufficient data.")
            continue
        keypoints_from_frame(episode, tf_data, reference_frame)

        # Validate all waypoint states
        # assert episode[-1]['keypoint']['status'] == 'success', f"Last waypoint is not success: {episode[-1]['keypoint']['status']}"
        # for i in range(1, len(episode)-1):
        #     assert episode[i]['keypoint']['status'] == 'running', f"Waypoint {i} is not progress: {episode[i]['keypoint']['status']}"
         
        # accumulate data to plot
        points, dir_pts = [], []
        for f in episode:
            # Note that the waypoints currently are represented in the body frame
            #   but we want body relative to the waypoint frame.
            pt, dir_pt = get_pose_details(f['keypoint']['pose'], perspective=PERSPECTIVE)
            points.append(pt); dir_pts.append(dir_pt)

        group.append([np.array(points), np.array(dir_pts), episode[-1]['keypoint']['status']])

    #### Plotting #######################################
    if SHOW_2D:
        # 2D
        fig = plt.figure()
        # plt.scatter(0, 0, c='g', alpha=0.25)
        # plt.plot([0, 0.2], [0, 0], c='g', alpha=0.25)
        legend = [p.name for p in bag_paths]
        for gi, (pts, dpts, status) in enumerate(group):
            plt.scatter(pts[:,0], pts[:,1], alpha=0.25)
            for pi, (p, frame) in enumerate(zip(pts, dpts)):
                x, y = [p[0], p[0]+frame[0]], [p[1], p[1]+frame[1]]
                plt.plot(x, y, c='r', alpha=0.25)                    
                if NUMBER_POINTS:
                    plt.annotate(f'{pi}', (p[0], p[1]))
            plt.plot(pts[:,0], pts[:,1], alpha=0.15, c='blue', label=legend[gi])
        plt.show()


    if SHOW_3D:
        # 3D  -  buggy
        for i, (pts, dpts, status) in enumerate(group):
            path = bag_paths[i]
            fig = plt.figure()
            points = np.array(pts)
            dir_pts = np.array(dpts)

            ax = fig.add_subplot(111, projection='3d')
            ax.title.set_text(f"{bag_paths[i]} \n {status}")        
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            
            # Robot
            ax.scatter(0, 0, 0, c='g', alpha=0.25)
            ax.plot([0, 0.2], [0, 0], [0, 0], c='g', alpha=0.25)

            # Points
            ax.scatter(points[:,0], points[:,1], points[:,2], alpha=0.25)
            
            # Directions
            for p, frame in zip(points, dir_pts):
                ax.plot([p[0], p[0]+frame[0]], [p[1], p[1]+frame[1]], [p[2], p[2]+frame[2]], c='r', alpha=0.25)
        plt.show()

    logging.getLogger(__name__).info('Done!')
