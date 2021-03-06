from open3d import *
import numpy as np

voxel_size = 0.02
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5


def load_point_clouds(voxel_size = 0.0):
    pcds = []
    for i in range(3):
        pcd = open3d.io.read_point_cloud("/home/rohit/Odometry/pcds/"+str(i)+".pcd")
        pcd_down = pcd.voxel_down_sample(voxel_size = voxel_size)
        pcds.append(pcd_down)
    return pcds


def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = open3d.pipelines.registration.registration_icp(source, target,
            max_correspondence_distance_coarse, np.identity(4),
            open3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = open3d.pipelines.registration.registration_icp(source, target,
            max_correspondence_distance_fine, icp_coarse.transformation,
            open3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = open3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine,
            icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds,
        max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = open3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(open3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                    pcds[source_id], pcds[target_id])
            print("Build PoseGraph")
            if target_id == source_id + 1: # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(open3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(open3d.pipelines.registration.PoseGraphEdge(source_id, target_id,
                        transformation_icp, information_icp, uncertain = False))
            else: # loop closure case
                pose_graph.edges.append(open3d.pipelines.registration.PoseGraphEdge(source_id, target_id,
                        transformation_icp, information_icp, uncertain = True))
    return pose_graph


if __name__ == "__main__":

    pcds_down = load_point_clouds(voxel_size)
    open3d.visualization.draw_geometries(pcds_down)

    print("Full registration ...")
    pose_graph = full_registration(pcds_down,
            max_correspondence_distance_coarse,
            max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = open3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance = max_correspondence_distance_fine,
            edge_prune_threshold = 0.25,
            reference_node = 0)
    open3d.pipelines.registration.global_optimization(pose_graph,
            GlobalOptimizationLevenbergMarquardt(),
            GlobalOptimizationConvergenceCriteria(), option)

    print("Transform points and display")
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    open3d.visualization.draw_geometries(pcds_down)

    print("Make a combined point cloud")
    pcds = load_point_clouds(voxel_size)
    pcd_combined = PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size = voxel_size)
    open3d.io.write_point_cloud("/home/rohit/Odometry/multiway_reg/multiway_registration.pcd", pcd_combined_down)
    open3d.visualization.draw_geometries([pcd_combined_down])
