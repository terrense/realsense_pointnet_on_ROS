#!/usr/bin/env python
import rospy
import torch
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from pointnet import PointNet

class PointNetROS:
    def __init__(self):
        self.pointnet = PointNet(num_classes=10)
        self.pointnet.load_state_dict(torch.load('pointnet_model.pth'))
        self.pointnet.eval()
        self.point_cloud_sub = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.callback)
        self.segmented_cloud_pub = rospy.Publisher("/segmented_cloud", PointCloud2, queue_size=10)

    def callback(self, data):
        points = np.array([p[:3] for p in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)])
        points = torch.from_numpy(points).float().unsqueeze(0)
        with torch.no_grad():
            predictions = self.pointnet(points)
        # Here you would process the predictions and publish the segmented point cloud
        # For simplicity, we just publish the original point cloud
        self.segmented_cloud_pub.publish(data)

if __name__ == '__main__':
    rospy.init_node('pointnet_node', anonymous=True)
    pn_ros = PointNetROS()
    rospy.spin()
