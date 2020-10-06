import numpy as np
import pickle
from eval_reconstruction import project
import argparse
import math
#import time



def expon(phi_3, mat):
    if phi_3.shape[0] != 3:
        print("ERROR SHAPE")
    else:
        #theta = np.arccos((np.trace(mat) -1)/2)
        x = np.linalg.norm(phi_3)
        ans = (np.identity(3)) + ((math.sin(x)/x) * mat) + ((((1- math.cos(x))/ (x**2)) * np.matmul(mat, mat)))
        return ans



def project2(P, pt, f=1):
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

    return (pt1[0], pt1[1], pt1[2], x, y)

def block_inv(mat, num):
    if mat.shape[0] % num != 0:
        raise ValueError
    start = 0
    new_ans = np.zeros((mat.shape[0], mat.shape[1]))
    block_arr = np.zeros((int(mat.shape[0]/ num),num, num ))
    #block_list = []

    for v in range(int(mat.shape[0]/ num)):
        v0 = v * num
        v1 = (v+1) * num
        b_mat = mat[v0: v1, v0: v1]
        #b_mat_inv = np.linalg.solve(b_mat, np.identity(b_mat.shape[0]))
        #new_ans[v0: v1, v0: v1] = b_mat_inv
        #block_list.append(b_mat)
        block_arr[v] = b_mat
    #block_arr = np.asarray(block_list)
    block_arr = np.linalg.inv(block_arr)
    for v in range(int(mat.shape[0]/ num)):
        v0 = v * num
        v1 = (v+1) * num
        new_ans[v0: v1, v0: v1] = block_arr[v]
    return new_ans, block_arr


def solve_ba_problem(problem):
    '''
    Solves the bundle adjustment problem defined by "problem" dict

    Input:
        problem: bundle adjustment problem containing the following fields:
            - is_calibrated: boolean, whether or not the problem is calibrated
            - observations: list of (cam_id, point_id, x, y)
            - points: [n_points,3] numpy array of 3d points
            - poses: [n_cameras,3,4] numpy array of camera extrinsics
            - focal_lengths: [n_cameras] numpy array of focal lengths
    Output:
        solution: dictionary containing the problem, with the following fields updated
            - poses: [n_cameras,3,4] numpy array of optmized camera extrinsics
            - points: [n_points,3] numpy array of optimized 3d points
            - (if is_calibrated==False) then focal lengths should be optimized too
                focal_lengths: [n_cameras] numpy array with optimized focal focal_lengths

    Your implementation should optimize over the following variables to minimize reprojection error
        - problem['poses']
        - problem['points']
        - problem['focal_lengths']: if (is_calibrated==False)

    '''

    solution = problem
    # YOUR CODE STARTS

    ## get names
    #for name,dict_ in solution.items():
        #print ('the name of the dictionary is ', name)

    is_cali = solution["is_calibrated"]
    focal_lengths = solution["focal_lengths"].copy()
    pts = solution["points"].copy()
    poses = solution["poses"].copy()
    observations = solution["observations"].copy()

    #print(is_cali)
    #print(focal_len)
    #print(points.shape)
    #print(poses.shape)
    #print(observations.shape)

    num_poses = poses.shape[0]
    num_pts = pts.shape[0]

    #print(poses.shape)
    #print(pts.shape)


    ## Compute the Jacobian



    #print(Jacobian.shape)

    tol=1e-5



    G_1 = np.asarray([[0,0,0],[0,0,-1],[0,1,0]])
    G_2 = np.asarray([[0,0,1],[0,0,0],[-1,0,0]])
    G_3 = np.asarray([[0,-1,0],[1,0,0],[0,0,0]])

    #print(G_1)
    #print(G_2)
    #print(G_3)
    avg_epe = 0.0
    First = True
    distrust = 1.5 ##1.2
    prev_loss = 0
    lr_init = 1
    lr = lr_init
    eta = 1
    decreased = True
    #eps = 0.000001
    if is_cali == True:
        foc_adj = 0
    else:
        foc_adj = 1


    #print(foc_adj)
    #print(is_cali)
    #print(len(observations))
    #print(2 * num_poses * num_pts)

    #intrinsic_uv = np.zeros((num_poses,2))

    i = 0
    while True:
        if decreased == True:
            avg_epe = 0.0
            J_T_J_s = np.zeros(((6 + foc_adj) * num_poses + 3 * num_pts,(6 + foc_adj) * num_poses + 3 * num_pts))

            J_T_e_s = np.zeros(((6 + foc_adj) * num_poses + 3 * num_pts, 1))
            #epe = 0.0
            #Jacobian = np.zeros((2 * num_poses * num_pts,(6 + foc_adj) * num_poses + 3 * num_pts))

            e_vals = np.zeros(2 * num_poses * num_pts)

            for j, (cam_id, pt_id, x_gt, y_gt) in enumerate(observations):
                #J_s = np.zeros((2,(6 + foc_adj) * num_poses + 3 * num_pts))

                #print(cam_id)
                #print(pt_id)
                #print(poses[cam_id].shape)
                R = poses[cam_id][0:3,0:3] ## Rotation matrix
                A_Mat = np.zeros((2,3))

                point = pts[pt_id]

                foc_len = focal_lengths[cam_id]

                x_prime, y_prime, z_prime, x_proj, y_proj =  project2(poses[cam_id], point, focal_lengths[cam_id])

                if foc_adj == 1:
                    f_mat = np.asarray([[x_prime/ z_prime],[y_prime / z_prime]])

                A_Mat[0,:] = [foc_len/ z_prime, 0, - (foc_len * x_prime) / (z_prime ** 2)]
                A_Mat[1,:] = [0, foc_len/ z_prime, - (foc_len * y_prime) / (z_prime ** 2)]

                point2 = (point[0] * G_1) + (point[1] * G_2) + (point[2] * G_3)

                G_Mat = - np.matmul(R, point2)

                B_Mat = np.matmul(A_Mat, G_Mat)

                #print(B_Mat)

                C_Mat = np.matmul(A_Mat, R)

                base = (6+foc_adj) * num_poses


                if foc_adj == 0:
                    J_mul = np.concatenate((B_Mat, A_Mat), axis =1)

                else:
                    Af_Mat = np.concatenate((A_Mat, f_mat), axis = 1)
                    J_mul = np.concatenate((B_Mat, Af_Mat), axis =1)

                #J_s[0:2,cam_id*(6+foc_adj):(cam_id+1)*(6+foc_adj)] = J_mul
                #J_s[0:2,base + (pt_id*3) : base + ((pt_id+1)*3)] = C_Mat
                J_T_J_s[cam_id*(6+foc_adj):(cam_id+1)*(6+foc_adj), cam_id*(6+foc_adj):(cam_id+1)*(6+foc_adj)] += (J_mul.T @ J_mul)
                J_T_J_s[base + (pt_id*3) : base + ((pt_id+1)*3), base + (pt_id*3) : base + ((pt_id+1)*3)] += (C_Mat.T @ C_Mat)

                #print(np.count_nonzero((J_mul.T @ J_mul)))

                e_v_s = np.asarray([[x_proj - x_gt], [y_proj - y_gt]])

                #J_T_e_s[cam_id*(6+foc_adj):(cam_id+1)*(6+foc_adj)] += J_mul.T @ e_v_s

                #J_T_e_s += J_s.T @ e_v_s

                J_T_e_s[cam_id*(6+foc_adj):(cam_id+1)*(6+foc_adj)] += J_mul.T @ e_v_s
                #print(C_Mat.T.shape)
                #print(e_v_s.shape)fstar
                J_T_e_s[base + (pt_id*3):base + ((pt_id+1)*3)] += C_Mat.T @ e_v_s


                #print(J_T_J_s)

                #print(J_T_e_s.shape)

                #x_proj, y_proj =  project(poses[cam_id], pts[pt_id], focal_lengths[cam_id])

                #e_vals[j*2: (j+1)*2] = [x_proj - x_gt, y_proj - y_gt]

                e_x = np.sum(np.square(e_v_s))

                epe = np.sqrt(e_x)
                avg_epe += epe / len(observations)#

            if avg_epe < 0.099: ## make the error
                #print("SIIIIIIIII LM ALGORITHM")
                break



        ## Update params

        #print(np.count_nonzero(e_vals))

        if decreased == False:
            distrust = distrust * 2
        elif decreased == True and First == False:
            distrust = distrust / 1.2

        #J_T_J = J_T_J_s s#np.matmul(Jacobian.T, Jacobian)
        #print(np.count_nonzero(J_T_J_s))
        J_T_J_diag_s = np.diag(np.diag(J_T_J_s))


        LHS = J_T_J_s + (distrust * J_T_J_diag_s) #+ (eps * np.identity(J_T_J.shape[0]))
        N = (6 + foc_adj) * num_poses
        M = 3 * num_pts
        A = LHS[0 : N,0 : N]
        C = LHS[0 : N, N: N + M]
        B = LHS[N : N + M, N : N+M]

        #schur = np.zeros((LHS.shape[0], LHS.shape[1]))

        #schur = np.identity(LHS.shape[0])

        #np.fill_diagonal(schur, 1)

        #B_inv_d = np.diag(1/ np.diag(B))

        #print(B[])

        #B_inv = np.linalg.solve(B, np.identity(B.shape[0]))

        B_inv, B_inv_block = block_inv(B, 3)
        #print(time.time() - start_a)



        C_block = C.reshape(C.shape[0], int(C.shape[1]/3), 3)
        C_block = np.transpose(C_block, (1,0,2))
        C_B_inv = np.einsum('ijk, ikl->ijl', C_block, B_inv_block)
        C_B_inv = np.transpose(C_B_inv, (1,0,2)).reshape(C.shape[0], C.shape[1])


        #C_B_inv = C @ B_inv

        #schur[0:N, N: N+M] = -C_B_inv

        RHS = - J_T_e_s

        #delta_x = np.linalg.solve(LHS, RHS)
        #RHS = schur @ RHS
        RHS[0:N] -= (C_B_inv @ RHS[N:N+M])

        LHS1 = A - (C_B_inv @ C.T)
        RHS1 = RHS[0:N]

        #print(RHS1.shape)

        delta_eps = np.linalg.solve(LHS1, RHS1)

        #print(delta_eps.shape)

        #LHS2 = B
        RHS2 = RHS[N: N+M] - (C.T @ delta_eps)


        delta_P = B_inv @ RHS2





        #print(delta_eps.shape)
        #print(delta_P.shape)

        #delta_x = np.concatenate((delta_eps, delta_P))


        ## UPDATES

        #print(poses.shape)


        for cam_id in range(num_poses):
            params = delta_eps[cam_id * (6+foc_adj): (cam_id+1)* (6+foc_adj)]
            #print(params.shape)
            phi_params = eta * params[0:3]
            phi_hat = (phi_params[0] * G_1) + (phi_params[1] * G_2) + (phi_params[2] * G_3)
            phi_expon = expon(phi_params, phi_hat)
            poses[cam_id][0:3,0:3] = np.matmul(poses[cam_id][0:3,0:3],phi_expon)
            #print(poses[cam_id][0:3,3].shape)
            #print(params[3:6].shape)
            poses[cam_id][0:3,3] += np.squeeze(params[3:6], axis = 1)
            if foc_adj ==1:
                focal_lengths[cam_id] += params[6]


        #print(pts.shape)

        delta_P = delta_P.reshape(pts.shape)

        pts += delta_P

        if decreased == False:
            avg_epe = 0.0
            for j, (cam_id, pt_id, x_gt, y_gt) in enumerate(observations):
                x_proj, y_proj =  project(poses[cam_id], pts[pt_id], focal_lengths[cam_id])


                e_x = (x_proj-x_gt)**2 + (y_proj-y_gt)**2

                epe = np.sqrt(e_x)
                avg_epe += epe
            avg_epe = avg_epe / len(observations)

        #if (i+1) % 5 == 0:
            #print("UPDATED")
            #print(i+1)
            #print(avg_epe)
            #print(distrust)
            #print(lr)

        if First == True:
            First = False
            prev_epe = avg_epe
            prev_epe2 = avg_epe
            poses2 = poses.copy() ##backedup
            pts2 = pts.copy() ##backedup
            if foc_adj == 1:
                foc2 = focal_lengths.copy()
            decreased = True

        elif avg_epe < prev_epe:
            if abs(avg_epe - prev_epe2) < 1e-10:  ##failsafe
                break
            prev_epe = avg_epe
            prev_epe2 = avg_epe

            #print("down")
            decreased = True
            poses2 = poses.copy() ##backedup
            pts2 = pts.copy() ##backedup
            if foc_adj == 1:
                foc2 = focal_lengths.copy()
            #current_decrease = Truefi


        else:
            #prev_epe = avg_epe
            prev_epe2 = avg_epe
            #print(avg_epe)
            #print(prev_epe)
            poses = poses2.copy()
            pts = pts2.copy()
            if foc_adj == 1:
                focal_lengths = foc2.copy()
            #print("up")
            decreased = False
        i +=1
        if (i+1) == 120: ## absolute cutoff
            break



    #print("SAVE ANS")

    solution["points"] = pts
    solution["poses"] = poses
    solution["focal_lengths"] = focal_lengths


















    #print(J_T_J_diag.shape)







    #mat_exp = np.linalg.matrix_power(np.zeros((3,3))




    return solution



if __name__ == '__main__':
    #t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', help="config file")
    args = parser.parse_args()

    problem = pickle.load(open(args.problem, 'rb'))
    solution = solve_ba_problem(problem)
    #t1 = time.time()
    #print("TIME ELAPSED")
    #print(t1-t0)
    solution_path = args.problem.replace(".pickle", "-solution.pickle")
    pickle.dump(solution, open(solution_path, "wb"))
