# SSDR Implementation in Python
# Dalton Omens

import maya.api.OpenMaya as om
import pymel.core as pm
import numpy as np
from scipy.optimize import lsq_linear
import time
import random


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
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    T = centroid_Q - centroid_P                       # translation vector
    P = P - centroid_P                                # Center both matrices on centroid
    Q = Q - centroid_Q
    H = np.dot(P.T, Q)                                # covariance matrix
    U, S, V = np.linalg.svd(H)                        # SVD
    d = (np.linalg.det(U) * np.linalg.det(V)) < 0.0   # correct rotation matrix for
    if d:                                             #  right-hand coordinate system
        V[:, -1] = -V[:, -1]
    R = np.dot(U, V).T                                # calculate optimal rotation
    return np.append(R, T)


def initialize(poses, rest_pose, num_bones, iterations=5):
    """
    Uses the k-means algorithm to initialize bone transformations.

    inputs: poses       |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose   |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            num_bones   Number of bones to initialize
            iterations  Number of iterations to run the k-means algorithm

    return: A |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked Rotation and Translation
              for each pose, for each bone.
    """
    num_verts = rest_pose.shape[0]
    num_poses = poses.shape[0]
    bone_transforms = np.empty((num_bones, num_poses, 4, 3))   # [(R, T) for for each pose] for each bone
                                                               # 3rd dim has 3 rows for R and 1 row for T
    vert_assignments = np.empty((num_verts))   # Bone assignment for each vertex

    # Randomly assign each vertex to a bone
    vert_assignments = np.random.randint(low=0, high=num_bones, size=num_verts)
    # Compute initial random bone transformations
    for bone in range(num_bones):
        for pose in range(num_poses):
            bone_transforms[bone, pose] = kabsch(rest_pose[vert_assignments == bone], poses[pose, vert_assignments == bone])
    
    for _ in range(iterations):
        # Re-assign bones to vertices using smallest reconstruction error from all poses
        # |num_bones| x |num_poses| x |num_verts| x 3
        Rp = np.dot(bone_transforms[:,:,:3,:], rest_pose.T).transpose((0, 1, 3, 2)) # R * p
        Rp_T = Rp + bone_transforms[:, :, np.newaxis, 3, :]  # R * p + T
        # |num_verts| x |num_bones| x |num_poses|
        errs_per_pose = np.linalg.norm(rest_pose - Rp_T , axis=3).transpose((2, 0, 1))
        vert_assignments = np.argmin(np.sum(errs_per_pose, axis=2), axis=1)

        # For each bone, for each pose, compute new transform using kabsch
        for bone in range(num_bones):
            for pose in range(num_poses):
                bone_transforms[bone, pose] = kabsch(rest_pose[vert_assignments == bone], poses[pose, vert_assignments == bone])

    return bone_transforms


def update_weight_map(bone_transforms, poses, rest_pose, sparseness):
    """
    Update the bone-vertex weight map W by fixing bone transformations and using a least squares
    solver subject to non-negativity constraint, affinity constraint, and sparseness constraint.

    inputs: bone_transforms |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
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
        Rp = np.dot(bone_transforms[:,:,:3,:], rest_pose[v]) # |num_bones| x |num_poses| x 3
        Rp_T = Rp + bone_transforms[:, :, 3, :]  # R * p + T
        A = Rp_T.transpose((1, 2, 0)).reshape((3 * num_poses, num_bones)) # 3 * |num_poses| x |num_bones|
        b = poses[:, v, :].reshape(3 * num_poses) # 3 * |num_poses| x 1

        # Bounds ensure non-negativity constraint and kind of affinity constraint
        w = lsq_linear(A, b, bounds=(0, 1), method='bvls').x

        w /= np.sum(w) # Ensure that w sums to 1 (affinity constraint)

        # TODO: Remove |B| - |K| bone weights with the least "effect"
        # how does argpartition work!!?!?

        W[v] = w

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

    bone_transforms = initialize(poses, rest_pose, num_bones)
    for _ in range(max_iterations):
        W = update_weight_map(bone_transforms, poses, rest_pose, sparseness)

    print(bone_transforms)
    
    end_time = time.time()
    print("Done. Calculation took {0} seconds".format(end_time - start_time))

# Get numpy vertex arrays from selected objects. Rest pose is most recently selected.
selectionLs = om.MGlobal.getActiveSelectionList()
num_poses = selectionLs.length() - 1
rest_pose = np.array(om.MFnMesh(selectionLs.getDagPath(num_poses)).getPoints())[:, :3]
poses = np.array([m.MFnMesh(selectionLs.getDagPath(num_poses)).getPoints() for i in range(num_poses)])[:, :, :3]

SSDR(poses, rest_pose, 2)