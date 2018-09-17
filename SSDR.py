# SSDR Implementation in Python
# Dalton Omens

import maya.api.OpenMaya as om
import pymel.core as pm
import numpy as np
from scipy.optimize import lsq_linear
from scipy.cluster.vq import vq, kmeans, whiten
import time


def kabsch(P, Q):
    """
    Computes the optimal translation and rotation matrices that minimize the 
    RMS deviation between two sets of points P and Q using Kabsch's algorithm.
    More here: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Inspiration: https://github.com/charnley/rmsd
    
    inputs: P  N x 3 numpy matrix representing the coordinates of the points in P
            Q  N x 3 numpy matrix representing the coordinates of the points in Q
            
    return: A 4 x 3 matrix where the first 3 rows are the rotation and the last is translation
    """
    if (P.size == 0 or Q.size == 0):
        raise ValueError("Empty matrices sent to kabsch")
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P                       # Center both matrices on centroid
    Q_centered = Q - centroid_Q
    H = np.dot(P.T, Q)                                # covariance matrix
    U, S, V = np.linalg.svd(H)                        # SVD
    R = np.dot(U, V).T                                # calculate optimal rotation
    if np.linalg.det(R) < 0:                          # correct rotation matrix for             
        V[2,:] *= -1                                  #  right-hand coordinate system
        R = np.dot(U, V).T                            
    t = centroid_Q - np.dot(R, centroid_P)            # translation vector
    return np.vstack((R, t))


def initialize(poses, rest_pose, num_bones, iterations=5):
    """
    Uses the k-means algorithm to initialize bone transformations.

    inputs: poses       |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose   |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            num_bones   Number of bones to initialize
            iterations  Number of iterations to run the k-means algorithm

    return: A |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked Rotation and Translation
              for each pose, for each bone.
            A |num_bones| x 3 matrix representing the translations of the rest bones.
    """
    num_verts = rest_pose.shape[0]
    num_poses = poses.shape[0]
    bone_transforms = np.empty((num_bones, num_poses, 4, 3))   # [(R, T) for for each pose] for each bone
                                                               # 3rd dim has 3 rows for R and 1 row for T
    vert_assignments = np.empty((num_verts))                   # Bone assignment for each vertex
    rest_bones_t = np.empty((num_bones, 3))                    # Translations for bones at rest pose
    rest_pose_corrected = np.empty((num_bones, num_verts, 3))  # Rest pose - mean of vertices attached to each bone

    # Randomly assign each vertex to a bone
    vert_assignments = np.random.randint(low=0, high=num_bones, size=num_verts)
    
    # Compute initial random bone transformations
    for bone in range(num_bones):
        rest_bones_t[bone] = np.mean(rest_pose[vert_assignments == bone], axis=0)
        rest_pose_corrected[bone] = rest_pose - np.mean(rest_pose[vert_assignments == bone], axis=0)
        for pose in range(num_poses):
            bone_transforms[bone, pose] = kabsch(rest_pose_corrected[bone, vert_assignments == bone], poses[pose, vert_assignments == bone])
    
    for it in range(iterations):
        # Re-assign bones to vertices using smallest reconstruction error from all poses
        constructed = np.empty((num_bones, num_poses, num_verts, 3)) # |num_bones| x |num_poses| x |num_verts| x 3
        for bone in range(num_bones):
            # |num_poses| x |num_verts| x 3
            Rp = np.dot(bone_transforms[bone,:,:3,:].transpose((0, 2, 1)), (rest_pose - rest_bones_t[bone]).T).transpose((0, 2, 1))
            constructed[bone] = Rp + bone_transforms[bone, :, np.newaxis, 3, :]  # R * p + t
        errs = np.linalg.norm(constructed - poses, axis=(1, 3))
        vert_assignments = np.argmin(errs, axis=0)    
        
        ## Visualization of vertex assignments for bone 0 over iterations
        ## Make 5 copies of an example pose mesh and call them test0, test1...
        #for i in range(num_verts):
        #    if vert_assignments[i] == 0:
        #        pm.select('test{0}.vtx[{1}]'.format(it, i), add=True)
        #print(vert_assignments)

        # For each bone, for each pose, compute new transform using kabsch
        for bone in range(num_bones):
            rest_bones_t[bone] = np.mean(rest_pose[vert_assignments == bone], axis=0)
            rest_pose_corrected[bone] = rest_pose - np.mean(rest_pose[vert_assignments == bone], axis=0)
            for pose in range(num_poses):
                bone_transforms[bone, pose] = kabsch(rest_pose_corrected[bone, vert_assignments == bone], poses[pose, vert_assignments == bone])

    return bone_transforms, rest_bones_t


def update_weight_map(bone_transforms, rest_bones_t, poses, rest_pose, sparseness):
    """
    Update the bone-vertex weight map W by fixing bone transformations and using a least squares
    solver subject to non-negativity constraint, affinity constraint, and sparseness constraint.

    inputs: bone_transforms |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
            rest_bones_t    |num_bones| x 3 matrix representing the translations of the rest bones
            poses           |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            sparseness      Maximum number of bones allowed to influence a particular vertex

    return: A |num_verts| x |num_bones| weight map representing the influence of the jth bone on the ith vertex
    """
    num_verts = rest_pose.shape[0]
    num_poses = poses.shape[0]
    num_bones = bone_transforms.shape[0]

    W = np.empty((num_verts, num_bones))

    for v in range(num_verts):
        # For every vertex, solve a least squares problem
        Rp = np.empty((num_bones, num_poses, 3))
        for bone in range(num_bones):
            # |num_bones| x |num_poses| x 3
            Rp[bone] = np.dot(bone_transforms[bone,:,:3,:], rest_pose[v] - rest_bones_t[bone]) 
        Rp_T = Rp + bone_transforms[:, :, 3, :]  # R * p + t
        A = Rp_T.transpose((1, 2, 0)).reshape((3 * num_poses, num_bones)) # 3 * |num_poses| x |num_bones|
        b = poses[:, v, :].reshape(3 * num_poses) # 3 * |num_poses| x 1

        # Bounds ensure non-negativity constraint and kind of affinity constraint
        w = lsq_linear(A, b, bounds=(0, 1), method='bvls').x  # |num_bones| x 1
        w /= np.sum(w) # Ensure that w sums to 1 (affinity constraint)

        # Remove |B| - |K| bone weights with the least "effect"
        effect = np.linalg.norm((A * w).reshape(num_poses, 3, num_bones), axis=1) # |num_poses| x |num_bones|
        effect = np.sum(effect, axis=0) # |num_bones| x 1
        num_discarded = max(num_bones - sparseness, 0)
        effective = np.argpartition(effect, num_discarded)[num_discarded:] # |sparseness| x 1

        # Run least squares again, but only use the most effective bones
        A_reduced = A[:, effective] # 3 * |num_poses| x |sparseness|
        w_reduced = lsq_linear(A_reduced, b, bounds=(0, 1), method='bvls').x # |sparseness| x 1
        w_reduced /= np.sum(w_reduced) # Ensure that w sums to 1 (affinity constraint)

        w_sparse = np.zeros(num_bones)
        w_sparse[effective] = w_reduced

        W[v] = w_sparse

    return W


def update_bone_transforms(W, bone_transforms, rest_bones_t, poses, rest_pose):
    """
    Updates the bone transformations by fixing the bone-vertex weight map and minimizing an
    objective function individually for each pose and each bone.
    
    inputs: W               |num_verts| x |num_bones| matrix: bone-vertex weight map. Rows sum to 1, sparse.
            bone_transforms |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
            rest_bones_t    |num_bones| x 3 matrix representing the translations of the rest bones
            poses           |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            
    return: |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
    """
    num_bones = W.shape[1]
    num_poses = poses.shape[0]
    num_verts = W.shape[0]
    
    for pose in range(num_poses):
        for bone in range(num_bones):
            # Calculate q_i for all vertices by equation (6)
            bone_transforms_strip = np.delete(bone_transforms, bone, axis=0)  # |num_bones-1| x |num_poses| x 4 x 3
            rest_bones_t_strip = np.delete(rest_bones_t, bone, axis=0)        # |num_bones-1| x 3
            Rp = np.empty((num_vertices, num_bones-1, 3))
            for bone2 in range(num_bones - 1): # maybe for vertices instead, so I can copy before? but I still have to do for all bones in the prev ugh
                
                p = rest_pose - rest_bones_t_strip[bone2]
                # |num_verts| x |num_bones-1| x 3
                Rp[:, bone2] = np.dot(bone_transforms_strip[bone2,pose,:3,:], p) 
            Rp_T = Rp + bone_transforms[:, :, 3, :]  # R * p + t
            q = poses[pose] - np.sum()

            # Calculate p_star, q_star, p_bar, and q_bar for all verts by equation (8)

            # Perform SVD by equation (9)

            # Calculate rotation R and translation t by equation (10)
    
    return None

def SSDR(poses, rest_pose, num_bones, sparseness=4, max_iterations=20):
    """
    Computes the Smooth Skinning Decomposition with Rigid bones
    
    inputs: poses           |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            num_bones       number of bones to create
            sparseness      max number of bones influencing a single vertex
            
    return: An i x j matrix of bone-vertex weights, where i = # vertices and j = # bones
            A length-t list of (length-B lists of bone transformations [R_j | T_j] ), one for each pose
            A list of the corrected vertex positions of the rest pose
    """
    start_time = time.time()

    bone_transforms, rest_bones_t = initialize(poses, rest_pose, num_bones)
    for _ in range(3):
        W = update_weight_map(bone_transforms, rest_bones_t, poses, rest_pose, sparseness)
        #bone_transforms = update_bone_transforms(W, bone_transforms, rest_bones_t, poses, rest_pose)

    print(bone_transforms)
    
    end_time = time.time()
    print("Done. Calculation took {0} seconds".format(end_time - start_time))

# Get numpy vertex arrays from selected objects. Rest pose is most recently selected.
selectionLs = om.MGlobal.getActiveSelectionList()
num_poses = selectionLs.length() - 1
rest_pose = np.array(om.MFnMesh(selectionLs.getDagPath(num_poses)).getPoints(om.MSpace.kWorld))[:, :3]
poses = np.array([om.MFnMesh(selectionLs.getDagPath(i)).getPoints(om.MSpace.kWorld) for i in range(num_poses)])[:, :, :3]

SSDR(poses, rest_pose, 2)