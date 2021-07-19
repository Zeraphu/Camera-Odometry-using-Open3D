import open3d as o3d
import os

if __name__ == "__main__":
    
    color_folder = '/home/rohit/Downloads/living_room_traj2_frei_png/rgb/'
    depth_folder = '/home/rohit/Downloads/living_room_traj2_frei_png/depth/'

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    pinhole_camera_intrinsic.set_intrinsics(640, 480, 481.20, -480.00, 319.50, 239.50)
    
    images = [img for img in os.listdir(color_folder) if img.endswith('.png')]
    
    for img_no in range(len(images)-879):
        source_color = o3d.io.read_image(color_folder + str(img_no) + '.png')
        source_depth = o3d.io.read_image(depth_folder + str(img_no) + '.png')
        source_rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(source_color, source_depth, convert_rgb_to_intensity = False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)
        
        print(img_no)
        o3d.io.write_point_cloud('/home/rohit/Odometry/'+ str(img_no) +'.ply', pcd)
