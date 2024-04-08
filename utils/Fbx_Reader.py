import sys
# from fbx import *
from FbxCommon import *

if __name__ == "__main__":
    fbx_path = '/usr/pydata/t2m/torch-mesh-isect/assets/fbx_file/motion_fix_assets/raw_motion_mabaoguo.fbx'
    sdk_manager, scene = InitializeSdkObjects()
    lResult = LoadScene(sdk_manager, scene, fbx_path)
    print('FBX')

# def ProcessOneFile(path, manager):
#     print('load file'+str(path))
#     LoadScene(manager,scene,path)
    