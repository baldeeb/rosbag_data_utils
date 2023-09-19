# Setup file for installing the package

from distutils.core import setup

setup(name='rosbag_utils',
        version='0.1',
        description='Utilities for translating rosbags to numpy arrays',
        author='Bahaa Aldeeb',
        author_email='aldeeb.bahaa@gmail.com',
        packages=['rosbag_utils'],
        requires=['rlbench', 'numpy', 'rospy', 'rosbag', 'tf2_ros', 
                  'geometry_msgs', 'sensor_msgs', 'std_msgs', 
                  'rospkg', 'pycryptodomex', 'gnupg', 'pyyaml',]
)

