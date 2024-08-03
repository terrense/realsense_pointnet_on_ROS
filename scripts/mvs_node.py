#!/usr/bin/env python
import rospy
import cv2
import pyrealsense2 as rs
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from mvs import generate_point_cloud
import sensor_msgs.point_cloud2 as pc2

class MVSNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.point_cloud_pub = rospy.Publisher("/camera/depth/color/points", PointCloud2, queue_size=10)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        self.color_image = None
        self.depth_image = None
        self.fx = 615.0  # Example focal length x
        self.fy = 615.0  # Example focal length y
        self.cx = 320.0  # Example principal point x
        self.cy = 240.0  # Example principal point y

    def image_callback(self, data):
        self.color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def depth_callback(self, data):
        self.depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        if self.color_image is not None:
            points = generate_point_cloud(self.color_image, self.depth_image, self.fx, self.fy, self.cx, self.cy)
            header = data.header
            point_cloud = pc2.create_cloud_xyz32(header, points)
            self.point_cloud_pub.publish(point_cloud)

if __name__ == '__main__':
    rospy.init_node('mvs_node', anonymous=True)
    mvs_node = MVSNode()
    rospy.spin()
