import numpy as np
import os
import sys
# import h5py
import cv2

# rot90y = np.array([[0, 0, -1],
#                    [0, 1, 0],
#                    [1, 0, 0]], dtype=np.float32)

def getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    F_MM = 35.  # Focal length
    SENSOR_SIZE_MM = 32.
    PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.
    SKEW = 0.
    CAM_MAX_DIST = 1.75
    CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                          [1.0, -4.371138828673793e-08, -0.0],
                          [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])

    # Calculate intrinsic matrix.
    # 2 atan(35 / 2*32)
    scale = RESOLUTION_PCT / 100
    # print('scale', scale)
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    # print('f_u', f_u, 'f_v', f_v)
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                           0,
                                           0)))
    # print('distance', distance_ratio * CAM_MAX_DIST)
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))

    return K, RT


def get_rotate_matrix(rotation_angle1):
    cosval = np.cos(rotation_angle1)
    sinval = np.sin(rotation_angle1)

    rotation_matrix_x = np.array([[1, 0, 0, 0],
                                  [0, cosval, -sinval, 0],
                                  [0, sinval, cosval, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_y = np.array([[cosval, 0, sinval, 0],
                                  [0, 1, 0, 0],
                                  [-sinval, 0, cosval, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_z = np.array([[cosval, -sinval, 0, 0],
                                  [sinval, cosval, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    scale_y_neg = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    neg = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    # y,z swap = x rotate -90, scale y -1
    # new_pts0[:, 1] = new_pts[:, 2]
    # new_pts0[:, 2] = new_pts[:, 1]
    #
    # x y swap + negative = z rotate -90, scale y -1
    # new_pts0[:, 0] = - new_pts0[:, 1] = - new_pts[:, 2]
    # new_pts0[:, 1] = - new_pts[:, 0]

    # return np.linalg.multi_dot([rotation_matrix_z, rotation_matrix_y, rotation_matrix_y, scale_y_neg, rotation_matrix_z, scale_y_neg, rotation_matrix_x])
    return np.linalg.multi_dot([neg, rotation_matrix_z, rotation_matrix_z, scale_y_neg, rotation_matrix_x])


def get_norm_matrix(sdf_h5_file):
    with h5py.File(sdf_h5_file, 'r') as h5_f:
        norm_params = h5_f['norm_params'][:]
        center, m, = norm_params[:3], norm_params[3]
        x, y, z = center[0], center[1], center[2]
        M_inv = np.asarray(
            [[m, 0., 0., 0.],
             [0., m, 0., 0.],
             [0., 0., m, 0.],
             [0., 0., 0., 1.]]
        )
        T_inv = np.asarray(
            [[1.0, 0., 0., x],
             [0., 1.0, 0., y],
             [0., 0., 1.0, z],
             [0., 0., 0., 1.]]
        )
    return np.matmul(T_inv, M_inv)


def get_W2O_mat(shift):
    T_inv = np.asarray(
        [[1.0, 0., 0., shift[0]],
         [0., 1.0, 0., shift[1]],
         [0., 0., 1.0, shift[2]],
         [0., 0., 0., 1.]]
    )
    return T_inv

def get_P_from_metadata(param):
    az, el, distance_ratio = param[0], param[1], param[3]
    K, RT = getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224)
    rot_mat = get_rotate_matrix(-np.pi / 2)
    W2O_mat = get_W2O_mat((param[-3], param[-1], -param[-2]))
    trans_mat = np.linalg.multi_dot([K, RT, rot_mat, W2O_mat])
    # trans_mat_right = np.transpose(trans_mat)
    return trans_mat

def gen_obj_img_h5():
    img_dir = "./test_render/image/03001627/17e916fc863540ee3def89b32cef8e45/hard/"
    sample_pc = np.asarray([[0, 0, 0]])
    colors = np.asarray([[0, 0, 255, 255]])
    # norm_mat = get_norm_matrix("/ssd1/datasets/ShapeNet/SDF_v2/03001627/17e916fc863540ee3def89b32cef8e45/ori_sample.h5")
    rot_mat = get_rotate_matrix(-np.pi / 2)
    for i in range(len(params[0])):
        param = params[0][i]
        camR, _ = get_img_cam(param)
        obj_rot_mat = np.dot(rot90y, camR)
        img_file = os.path.join(img_dir, '{0:02d}.png'.format(i))
        out_img_file = os.path.join(img_dir, '{0:02d}_out.png'.format(i))
        print("img_file", img_file)
        img_arr = cv2.imread(img_file, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        az, el, distance_ratio = param[0], param[1], param[3]
        K, RT = getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224)
        print((-param[-1], -param[-2], param[-3]))
        W2O_mat = get_W2O_mat((param[-3], param[-1], -param[-2]))
        # trans_mat = np.linalg.multi_dot([K, RT, rot_mat, W2O_mat, norm_mat])
        trans_mat = np.linalg.multi_dot([K, RT, rot_mat, W2O_mat])
        trans_mat_right = np.transpose(trans_mat)
        # regress_mat = np.transpose(np.linalg.multi_dot([RT, rot_mat, W2O_mat, norm_mat]))
        pc_xy = get_img_points(sample_pc, trans_mat_right)  # sample_pc - camloc
        print("trans_mat_right", trans_mat_right)
        for j in range(pc_xy.shape[0]):
            y = int(pc_xy[j, 1])
            x = int(pc_xy[j, 0])
            print(x,y)
            cv2.circle(img_arr, (x, y), 10, tuple([int(x) for x in colors[j]]), -2)
        cv2.imwrite(out_img_file, img_arr)


def get_img_points(sample_pc, trans_mat_right):
    sample_pc = sample_pc.reshape((-1, 3))
    homo_pc = np.concatenate((sample_pc, np.ones((sample_pc.shape[0], 1), dtype=np.float32)), axis=-1)
    pc_xyz = np.dot(homo_pc, trans_mat_right).reshape((-1, 3))
    # pc_xyz = np.transpose(np.matmul(trans_mat, np.transpose(homo_pc))).reshape((-1,3))
    # print("pc_xyz",pc_xyz)
    print("pc_xyz shape: ", pc_xyz.shape)
    pc_xy = pc_xyz[:, :2] / pc_xyz[:, 2]
    return pc_xy.astype(np.int32)


def get_img_cam(param):
    cam_mat, cam_pos = camera_info(degree2rad(param))

    return cam_mat, cam_pos


def camera_info(param):
    az_mat = get_az(param[0])
    el_mat = get_el(param[1])
    inl_mat = get_inl(param[2])
    cam_mat = np.transpose(np.matmul(np.matmul(inl_mat, el_mat), az_mat))
    cam_pos = get_cam_pos(param)
    return cam_mat, cam_pos


def get_cam_pos(param):
    camX = 0
    camY = 0
    camZ = param[3]
    cam_pos = np.array([camX, camY, camZ])
    return -1 * cam_pos


def get_az(az):
    cos = np.cos(az)
    sin = np.sin(az)
    mat = np.asarray([cos, 0.0, sin, 0.0, 1.0, 0.0, -1.0 * sin, 0.0, cos], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat


def get_el(el):
    cos = np.cos(el)
    sin = np.sin(el)
    mat = np.asarray([1.0, 0.0, 0.0, 0.0, cos, -1.0 * sin, 0.0, sin, cos], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat


def get_inl(inl):
    cos = np.cos(inl)
    sin = np.sin(inl)
    # zeros = np.zeros_like(inl)
    # ones = np.ones_like(inl)
    mat = np.asarray([cos, -1.0 * sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat


def degree2rad(params):
    params_new = np.zeros_like(params)
    params_new[0] = np.deg2rad(params[0] + 180.0)
    params_new[1] = np.deg2rad(params[1])
    params_new[2] = np.deg2rad(params[2])
    return params_new

