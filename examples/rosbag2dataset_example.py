# add parent path to sys.path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from data_preprocessing.rosbag_as_dataset import get_topics_in_path, RosbagReader

if __name__ == '__main__':
    '''This is an example of how to use the RosbagReader class.'''
    import matplotlib.pyplot as plt

    # Specify bag path
    bag_path = '/media/baldeeb/ssd2/Data/kinect/images_poses_camerainfo.bag'

    # Determine the topics you want to acquire and their desired names
    print(get_topics_in_path(bag_path))
    topic_names = {'/rtabmap/rtabmap/localization_pose': 'camera_pose',
                   '/k4a/depth_to_rgb/camera_info': 'intrinsics',
                   '/k4a/depth_to_rgb/image_raw': 'depth',
                   '/k4a/rgb/image_raw': 'color',}
    
    # Create the dataset with this information
    dataset = RosbagReader(bag_path, topics_and_names=topic_names)
    
    # Voila!
    for name, data in dataset[0].items():
        try:
            plt.title(name); plt.imshow(data); plt.show()
        except:
            print(name)

    # NOTE: You can also register your own postprocessing functions
    # By default: 
    #   - any image type will be processed to numpy.
    #   - any pose with covariance will be processed to a 4x4 matrix.

if __name__ == '__main__':
    '''This is an example of how to use the RosbagReader class.'''
    import matplotlib.pyplot as plt

    # Specify bag path
    bag_path = '/home/rpm/revo/new_setup/ros_ws/src/data/example.bag'

    # Determine the topics you want to acquire and their desired names
    print(get_topics_in_path(bag_path))
    topics_and_names = {'/spot/camera/frontleft/camera_info': 'frontleft_camera_info',
                    '/spot/camera/frontright/camera_info': 'frontright_camera_info',
                    '/spot/camera/frontleft/image': 'frontleft_rgb',
                    '/spot/camera/frontright/image': 'frontright_rgb',
                    '/spot/depth/frontleft/image': 'frontleft_depth',
                    '/spot/depth/frontright/image': 'frontright_depth',
                    '/spot/odometry': 'pose'
                    }

    # Create the dataset with this information

    bag = rosbag.Bag(bag_path)

    dataset = RosbagReader(bag_path, bag, topic_names)

    # Voila!
    # for name, data in dataset[10].items():
    #     try:
    #         plt.title(name); plt.imshow(data); plt.show()
    #     except:
    #         print(name)

    # print(list(dataset)[20])

    # with open('spot_data.pkl', 'wb') as fp:
    #     pickle.dump(dataset, fp)
    #     print('dictionary saved successfully to file')

    # NOTE: You can also register your own postprocessing functions
    # By default: 
    #   - any image type will be processed to numpy.
    #   - any pose with covariance will be processed to a 4x4 matrix.