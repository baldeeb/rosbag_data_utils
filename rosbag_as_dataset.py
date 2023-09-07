import rospy
import rosbag
from ros_to_numpy_helpers import (
    ros_camera_info_to_np_intrinsic,
    ros_image_to_np,
    ros_pose_to_np_se3_matrix
)
import logging

def get_topics_in_path(path:str):
    bag = rosbag.Bag(path)
    return list(bag.get_type_and_topic_info()[1])

class MissingTopicError(Exception):
    def __init__(self, message, topics):
        super().__init__(message)
        self.topics = topics

class RosbagReader:
    ''' 
    Takes a ros bag and returns one item at a time.
    Assumes that the bag topics are logged in a normally distributed fashion. 
    ''' 
    def __init__(self, 
                 path, 
                 topics_and_names=None, 
                 default_topic_process=True, 
                 anytime_topics=[],
                 permissible_asynchronisity_sec=None,
                 reference_topic=None):
        '''
        Args:
            path: path to the rosbag file
            topics_and_names: a dictionary of topic names to be used in the output
            default_topic_process: if True, the reader will use default postprocess functions
        '''
        self.bag = rosbag.Bag(path)

        if topics_and_names is not None:
            available_topics = self.get_topics()
            desired_topics = list(topics_and_names.keys())
            self._validate_requested_topics(desired_topics, available_topics)
            self._topics = desired_topics
        else:
            self._topics = self.get_topics()

        # Select reference topic
        if reference_topic is not None:
            self._len, self._time_slots = self._get_reference_topic_info(reference_topic)
        else:
            reference_topic, self._len, self._time_slots = self._get_topic_with_least_messages()
        
        slack_sec = self._resolve_timing_details(permissible_asynchronisity_sec)
        self._resolve_naming_details(topics_and_names)

        self._anytime_topics = anytime_topics

        self._postprocess_type = {}
        self._postprocess_topic = {}
        self._postprocess_output_name = {}

        if default_topic_process:
            self._postprocess_type = {
                'sensor_msgs/Image': ros_image_to_np,
                'geometry_msgs/PoseWithCovarianceStamped': ros_pose_to_np_se3_matrix,
                'sensor_msgs/CameraInfo': ros_camera_info_to_np_intrinsic,
            }

        logging.info(f'Initialized BagReader - ref topic: {reference_topic} - time slack {slack_sec}.')


    def _resolve_timing_details(self, slack_sec=None):
        '''Determines time window around said topic's messages to search for other topics'''
        self._start, self._end = self.bag.get_start_time(), self.bag.get_end_time()
        if slack_sec is not None: 
            self._time_eps = rospy.Time.from_sec(slack_sec)
        else: 
            slack_sec = (self._end - self._start) / self.__len__()
            self._time_eps = rospy.Time.from_sec(slack_sec)
        return slack_sec

    def _resolve_naming_details(self, topics_and_names):
        '''Figures out what names to use for what topics in output.'''
        if topics_and_names is not None:
            self._name_of_topic = topics_and_names
        else: 
            self._name_of_topic = {}
            for t in self._topics:
                if t not in self._name_of_topic:
                    self._name_of_topic[t] = t

    def _validate_requested_topics(self, desired, available):
        if not all([t in available for t in desired]):
            not_found = [t for t in desired if t not in available]
            raise RuntimeError(f"The topics {not_found} are not available in the bag.\n" +
                               f'available topics: {available}')    
    
    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]


    def clear_postprocess(self):
        self._postprocess_type = {}
    def unregister_type_postprocess(self, msg_type:str):
        del self._postprocess_type[msg_type]
    def register_type_postprocess(self, msg_type:str, func:callable):
        self._postprocess_type[msg_type] = func


    def register_topic_postprocess(self, topic:str, func:callable):
        self._postprocess_topic[topic] = func
    def unregister_topic_postprocess(self, topic:str):
        del self._postprocess_topic[topic]


    def register_output_postprocess(self, msg_name:str, func:callable):
        self._postprocess_output_name[msg_name] = func
    def unregister_output_postprocess(self, msg_name:str):
        del self._postprocess_output_name[msg_name]


    def __len__(self): return self._len


    def _get_topic_with_least_messages(self):
        topic = self._topics[0]
        count = self.bag.get_message_count(topic)
        for t in self._topics[1:]:
            c = self.bag.get_message_count(t)
            if c < count: count, topic = c, t
        time_sec = [m[2] for m in self.bag.read_messages(topics=[topic])]
        return topic, count, time_sec  

    def _get_reference_topic_info(self, ref):
        count = self.bag.get_message_count(ref)
        time_sec = [m[2] for m in self.bag.read_messages(topics=[ref])]
        return count, time_sec  


    def get_topics(self):
        return list(self.bag.get_type_and_topic_info()[1])


    def _msgs_at(self, time:rospy.Time, rigourous=True, anytime=False, topics=None):
        '''Ensures that messages are returned by proximity from intended time.'''
        if topics is None: topics = self._topics
        if anytime: 
            rigourous = True
            s = e = None
        else:
            t_time, elapsed = time.to_time(), self._time_eps.to_time()
            s = rospy.Time.from_sec(t_time - elapsed)
            e = rospy.Time.from_sec(t_time + elapsed)

        msgs = self.bag.read_messages(topics = topics, 
                                      start_time = s, 
                                      end_time = e)
        if rigourous:
            msgs = list(msgs)
            dt = [abs(m[2] - time) for m in msgs]
            idxs = sorted(range(len(dt)), key=lambda k: dt[k])
            for i in idxs:
                yield msgs[i]
        else:
            for m in msgs:
                yield m


    def _process(self, topic, name, msg):
        data = {}
        if topic in self._postprocess_topic:           # if topic has a postprocess function
            data[name] = self._postprocess_topic[topic](msg)
        elif msg._type in self._postprocess_type:    # if message type has a postprocess function
            data[name] = self._postprocess_type[msg._type](msg)
        else:                                           # no postprocess function
            data[name] = msg

        if name in self._postprocess_output_name:   # if generated data has a postprocess function
            data[name] = self._postprocess_output_name[name](data[name])
        return data


    def __getitem__(self, idx):
        time = self._time_slots[idx]
        data = {}
        for msg in self._msgs_at(time):
            msg_name = self._name_of_topic[msg[0]]
            if msg_name in data: continue                           # Continue, if topic already obtained
            data.update(self._process(msg[0], msg_name, msg[1]))    # Add msg to data
            if len(data) == len(self._topics): break                # All topics collected
        
        # If some topics are set to time agnostic, try to get them from anytime
        if len(data) != len(self._topics):
            if len(self._anytime_topics) > 0:
                missing = []
                for topic, name in self._name_of_topic.items():
                    if name not in data:
                        if name in self._anytime_topics:
                            msg = next(iter(self._msgs_at(time, anytime=True, 
                                                        topics=[topic])))[1]
                            data.update(self._process(topic, name, msg))
                        else:
                            missing.append(name)
                if len(missing) > 0:
                    raise MissingTopicError(f'Not all topics were found at time {time}.', 
                                            topics=missing)
            else:
                raise MissingTopicError(f'Not all topics were found at time {time}.', 
                                        topics=list(self._name_of_topic.values()))
        return data

    def __del__(self):
        self.bag.close()
