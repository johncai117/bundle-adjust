import numpy as np
import argparse
import pickle


def project(P, pt, f=1):
    '''
    Project a 3D point onto the image
    Input:
        P: camera extrinsics matrix, dimension [3, 4]
        pt: 3D points in world coordinates pt=[X, Y, Z]
        f: focal length
    Output:
        (x,y): image point
    '''

    pt1 = np.dot(P, [pt[0], pt[1], pt[2], 1.0])
    x = f*(pt1[0]/pt1[2])
    y = f*(pt1[1]/pt1[2])

    return (x, y)



def reprojection_error(problem, tol=1e-5):
    '''
    Evaluate the final reconstruction on average reprojection error
    Input:
        problem: bundle adjustment problem containing the following fields:
            - observations: list of (cam_id, point_id, x, y)
            - points: [n_points,3] numpy array of 3d points
            - poses: [n_cameras,3,4] numpy array of camera extrinsics
            - focal_lengths: [n_cameras] numpy array of focal lengths
    Output:
        avg_epe: average reprojection error over all observations
    '''

    focal_lengths = problem['focal_lengths']
    observations = problem['observations']
    poses = problem['poses']
    pts = problem['points']

    avg_epe = 0.0
    for (cam_id, pt_id, x_gt, y_gt) in observations:
        R = poses[cam_id][0:3,0:3]

        if np.abs(np.linalg.det(R) - 1.0) > tol:
            print("camera %d does not have a valid rotation matrix" % cam_id)
            avg_epe = -1
            break

        if np.max(np.abs(np.eye(3)-np.dot(R.T,R))) > tol:
            print("camera %d does not have a valid rotation matrix" % cam_id)
            avg_epe = -1
            break

        x_proj, y_proj =  project(poses[cam_id], pts[pt_id], focal_lengths[cam_id])
        epe = np.sqrt((x_proj-x_gt)**2 + (y_proj-y_gt)**2)
        avg_epe += epe / len(observations)

    print("Average Reprojection Error: %f" % avg_epe)
    return avg_epe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution', help="config file")
    args = parser.parse_args()

    solution = pickle.load(open(args.solution, 'rb'))
    reprojection_error(solution)
