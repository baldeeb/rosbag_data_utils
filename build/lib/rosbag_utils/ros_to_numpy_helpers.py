from sensor_msgs.msg import (
    Image, 
    CameraInfo,
    )
from geometry_msgs.msg import (
    Pose,
    Point, 
    Quaternion,
    PoseWithCovarianceStamped,
    Transform
    )
import numpy as np
from quaternion import as_rotation_matrix, quaternion, from_rotation_matrix


def ros_camera_info_to_np_intrinsic(info:CameraInfo):
    return np.array([
        [info.K[0], info.K[1], info.K[2]],
        [info.K[3], info.K[4], info.K[5]],
        [info.K[6], info.K[7], info.K[8]]
    ])

def ros_image_to_np(image:Image, depth_to_meters=1e-3):
    H, W = image.height, image.width
    if image.encoding == 'rgb8':
        rgb = np.frombuffer(image.data, dtype=np.byte)
        img = rgb.reshape(H, W, 3).astype(np.uint8)
    elif image.encoding == 'rgba8':
        rgb = np.frombuffer(image.data, dtype=np.byte)
        img = rgb.reshape(H, W, 4).astype(np.uint8)
    elif image.encoding == 'bgra8':
        rgb = np.frombuffer(image.data, dtype=np.byte)
        img = rgb.reshape(H, W, 4)[:, :, (2,1,0)].astype(np.uint8)
    elif image.encoding == '16UC1':
        d = np.frombuffer(image.data, dtype=np.uint16).reshape(H, W)
        img = d.astype(np.float32) * depth_to_meters
    elif image.encoding == 'bgra8':
        rgbd = np.frombuffer(image.data, dtype=np.byte)
        rgbd = rgbd.reshape(H, W, 4)[:, :, (2,1,0)].astype(np.uint8)
        img = rgbd[:,:,3].astype(np.uint16).astype(np.float32) * depth_to_meters
    else: 
        raise RuntimeError(f'Image to Numpy is not setup to handle {image.encoding}.')
    return img

def _ros_point_to_np(point:Point):
    return np.array([point.x, point.y, point.z])

def _ros_quaternion_to_np(quat:Quaternion):
    q = quaternion(quat.w, quat.x, quat.y, quat.z)
    return as_rotation_matrix(q)

def ros_pose_to_np_se3_matrix(pose:Pose):
    mat = np.eye(4)
    mat[:3,:3] = _ros_quaternion_to_np(pose.orientation)
    mat[:3,3] = _ros_point_to_np(pose.position)
    return mat

def ros_pose_to_7d_position_quaternion(pose:Pose):
    vec = np.empty(7)
    vec[:3] = _ros_point_to_np(pose.position)
    vec[3:] = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    return vec
    
def ros_pose_with_cov_stamped_to_np_se3_matrix(pose:PoseWithCovarianceStamped):
    return ros_pose_to_np_se3_matrix(pose.pose.pose)

def ros_transform_to_np_se3_matrix(transform:Transform):
    mat = np.eye(4)
    mat[:3,:3] = _ros_quaternion_to_np(transform.rotation)
    mat[:3,3] = _ros_point_to_np(transform.translation)
    return mat

def se3_mat_to_position_and_quaterion_vec(mat):
    vec = np.empty(7)
    vec[:3] = mat[:3,3]
    q = from_rotation_matrix(mat[:3,:3])
    vec[3:] = np.array([q.x, q.y, q.z, q.w]) 
    return vec

def ros_joint_states_to_numpy(msg):
    return {
        'names': msg.name,
        'positions': np.array(msg.position),
        'velocities': np.array(msg.velocity),
        'efforts': np.array(msg.effort),
    }

