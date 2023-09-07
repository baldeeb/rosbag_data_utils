from rospy import Time
from data_preprocessing.ros_tf2_wrapper import get_populated_tf2_wrapper, add_odometry_to_tf2wrapper
if __name__ == '__main__':
    path = '/media/baldeeb/ssd2/Data/SpotBags/30_06_2023/move_to_right_01_2023-06-30-21-07-51.bag'

    # Fill the buffer using a rosbag
    dynamic_topics =['/tf'] 
    static_topics = ['/tf_static']
    tf_wrapper = get_populated_tf2_wrapper(path, static_topics, dynamic_topics)
    # add_odometry_to_tf2wrapper(path, '/spot/odometry', tf_wrapper)
    print(tf_wrapper)

    # Get data from the buffer
    t = tf_wrapper.get('body', 'rear_right_upper_leg', Time(1688159282, 693))
    print(t)