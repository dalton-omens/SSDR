# SSDR
## Implementation of Le and Deng's paper "Smooth Skinning Decomposition with Rigid Bones" in Maya

This is a Python script which is meant to run within Maya. This means that only Python 2.7 is supported at this time.

For help with installing numpy and scipy in Maya's python interpreter, I found this link very helpful:
https://forums.autodesk.com/t5/maya-programming/guide-how-to-install-numpy-scipy-in-maya-windows-64-bit/td-p/5796722

## How to use
To run the script out of the box, first you must have any number of deformations of the same mesh topology, for example different poses of a character.

Then select all of the meshes (and only the meshes), with the "rest pose" selected last.

Now run the script and it should calculate a weight map, bone transforms for each pose, and the translations of the bones at the rest position.

## Other info
This is not an exact recreation of the algorithm in Le and Deng's paper.

The "rest bones" is not a concept explicitly described in the paper but helps converge to a better solution. To get to a final mesh from the result bones, we first subtract off the rest bone translation for every bone and every vertex, then do the standard W^T * (R * p + T) tansformation to get to the final pose. This is demonstrated in the reconstruct function with the additional pose dimension.

We also do not preform bone re-initialization.
