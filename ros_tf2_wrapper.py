from geometry_msgs.msg import TransformStamped 
from std_msgs.msg import Time
from tf2_ros import Buffer, TransformException
from rospy import Duration
from typing import List, Optional
from rosbag_as_dataset import RosbagReader
from ros_to_numpy_helpers import ros_transform_to_np_se3_matrix
import yaml

class Tf2Wrapper:
    def __init__(self, 
                 authority:str='tf2_wrapper', 
                 cache_time:Optional[Duration]=None
                 ):
        self._buffer = Buffer(cache_time, debug=False)
        self._authority = authority
        self._hooks = {}

    def register_getter_hook(self, hook, ref_frame:str=None, frame:str=None):
        if ref_frame is not None and frame is not None:
            self._hooks[(ref_frame, frame)] = hook
        elif ref_frame is not None:
            self._hooks[ref_frame] = hook
        elif frame is not None:
            self._hooks[frame] = hook
        else:
            raise ValueError('Either ref_frame or frame must be specified')

    def set(self, transform:TransformStamped, static:bool):
        if static:
            self._buffer.set_transform_static(transform, self._authority)
        else:
            self._buffer.set_transform(transform, self._authority)

    def _post_process(self, T, ref_frame, frame):
        if (ref_frame, frame) in self._hooks:
            return self._hooks[(ref_frame, frame)](T)
        elif ref_frame in self._hooks:
            return self._hooks[ref_frame](T)
        elif frame in self._hooks:
            return self._hooks[frame](T)
        else:
            return T

    def get(self, reference:str, frame_name:str, timestamp:Time):
        t = self._buffer.lookup_transform(reference, frame_name, timestamp)
        T = ros_transform_to_np_se3_matrix(t.transform)
        T = self._post_process(T, reference, frame_name)
        return T
    
    def __str__(self):
        return self._buffer.all_frames_as_yaml()
    
    def get_as_dict(self):
        return yaml.safe_load(self._buffer.all_frames_as_yaml())


def get_populated_tf2_wrapper(bag_path:str, 
                              static_topics:List[str]= ['/tf_static'], 
                              dynamic_topics:List[str]=['/tf'],
                              cache_time_sec:float=60*2,
                              slack_sec:float=0.1,
                              ):

    def add_topics(topics, is_static):
        for topic in topics:
            dataset = RosbagReader(bag_path, {topic: 't'}, 
                                   permissible_asynchronisity_sec=slack_sec)
            for d in dataset:
                for transfrom in d['t'].transforms:
                    wrapper.set(transfrom, is_static)
    
    wrapper = Tf2Wrapper(cache_time=Duration(cache_time_sec))
    add_topics(static_topics, True)
    add_topics(dynamic_topics, False)

    return wrapper
