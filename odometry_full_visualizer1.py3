import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    
    color_folder = '/home/rohit/Downloads/living_room_traj2_frei_png/rgb/'
    depth_folder = '/home/rohit/Downloads/living_room_traj2_frei_png/depth/'
    
    cam = o3d.camera.PinholeCameraParameters()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    pinhole_camera_intrinsic.set_intrinsics(640, 480, 481.20, -480.00, 319.50, 239.50) #481.20, -480.00, 319.50, 239.50
    cam.intrinsic = pinhole_camera_intrinsic
    print(cam.intrinsic.intrinsic_matrix)
    
    images = [img for img in os.listdir(color_folder) if img.endswith('.png')]
    
    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    print(option)
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    
    target_color = o3d.io.read_image(color_folder + '1.png')
    target_depth = o3d.io.read_image(depth_folder + '1.png')
    source_color = o3d.io.read_image(color_folder + '0.png')
    source_depth = o3d.io.read_image(depth_folder + '0.png')
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(source_color, source_depth, convert_rgb_to_intensity = False)
    source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, cam.intrinsic)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(target_color, target_depth, convert_rgb_to_intensity = False)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, cam.intrinsic)
    vis.add_geometry(source_pcd_hybrid_term)
     
    for img_no in range(len(images)-1):
        
        source_color = o3d.io.read_image(color_folder + str(img_no) + '.png')
        source_depth = o3d.io.read_image(depth_folder + str(img_no) + '.png')
        target_color = o3d.io.read_image(color_folder + str(img_no+1) + '.png')
        target_depth = o3d.io.read_image(depth_folder + str(img_no+1) + '.png')
    
        source_rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(source_color, source_depth, convert_rgb_to_intensity = False)
        target_rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(target_color, target_depth, convert_rgb_to_intensity = False)
        target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, cam.intrinsic)
    # Debug presentations
        #target_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #o3d.visualization.draw_geometries([target_pcd])
        #plt.imshow(target_rgbd_image.color)
        #plt.show()
    
        [success_hybrid_term, trans_hybrid_term, info] = o3d.pipelines.odometry.compute_rgbd_odometry(source_rgbd_image, target_rgbd_image, 
                                           cam.intrinsic, odo_init, o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
     
        if success_hybrid_term:
            print(img_no)
            source_pcd_hybrid_term.transform(trans_hybrid_term)
            
            view_ctl = vis.get_view_control()
            final = np.sum(target_pcd, source_pcd_hybrid_term)
            vis.update_geometry(final)
            if img_no%10 == 0:
                vis.update_renderer()
            
            cam = view_ctl.convert_to_pinhole_camera_parameters()
            cam.extrinsic = trans_hybrid_term
            
            view_ctl.convert_from_pinhole_camera_parameters(cam)
            vis.run()

    o3d.io.write_point_cloud('/home/rohit/Odometry/final.ply', source_pcd_hybrid_term)















