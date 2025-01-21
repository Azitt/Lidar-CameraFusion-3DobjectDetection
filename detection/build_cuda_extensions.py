import os
from typing import List
from torch.utils import cpp_extension

IOU_3D_NMS_CUDA_SOURCES = [
                        'CUDA_extension_modules/iou3d_nms_cuda/iou3d_cpu.cpp',
                        'CUDA_extension_modules/iou3d_nms_cuda/iou3d_nms_api.cpp',
                        'CUDA_extension_modules/iou3d_nms_cuda/iou3d_nms.cpp',
                        'CUDA_extension_modules/iou3d_nms_cuda/iou3d_nms_kernel.cu',
                    ]

ROI_AWARE_POOL3D_SOURCES = [
                        'CUDA_extension_modules/roiaware_pool3d/roiaware_pool3d.cpp',
                        'CUDA_extension_modules/roiaware_pool3d/roiaware_pool3d_kernel.cu',
                    ]

POINTNET2_STACK_SOURCES = [
                            'CUDA_extension_modules/pointnet2_stack/pointnet2_api.cpp',
                            'CUDA_extension_modules/pointnet2_stack/ball_query.cpp',
                            'CUDA_extension_modules/pointnet2_stack/ball_query_gpu.cu',
                            'CUDA_extension_modules/pointnet2_stack/group_points.cpp',
                            'CUDA_extension_modules/pointnet2_stack/group_points_gpu.cu',
                            'CUDA_extension_modules/pointnet2_stack/sampling.cpp',
                            'CUDA_extension_modules/pointnet2_stack/sampling_gpu.cu', 
                            'CUDA_extension_modules/pointnet2_stack/interpolate.cpp', 
                            'CUDA_extension_modules/pointnet2_stack/interpolate_gpu.cu',
                            'CUDA_extension_modules/pointnet2_stack/voxel_query.cpp', 
                            'CUDA_extension_modules/pointnet2_stack/voxel_query_gpu.cu',
                            'CUDA_extension_modules/pointnet2_stack/vector_pool.cpp',
                            'CUDA_extension_modules/pointnet2_stack/vector_pool_gpu.cu'                   
                    ]

def buildCUDAExtension(moduleName : str, buildDir : str, rootDir : str, sourceFiles : List[str]):
    return cpp_extension.load(name = moduleName, verbose=False, build_directory=buildDir,
                  sources = [os.path.join(rootDir, x) for x in sourceFiles])
    

def buildIou3dNmsModule(rootDir : str, buildDir : str, name : str = "iou3d_nms_cuda"):
    return buildCUDAExtension(moduleName=name, buildDir=buildDir, rootDir=rootDir, 
                              sourceFiles=IOU_3D_NMS_CUDA_SOURCES)

def buildPointNet2Stack(rootDir : str, buildDir : str, name : str = "pointnet2_stack_cuda"):
    return buildCUDAExtension(moduleName=name, buildDir=buildDir, rootDir=rootDir, 
                              sourceFiles=POINTNET2_STACK_SOURCES)

def buildRoiAwarePool3d(rootDir : str, buildDir : str, name : str = "roiaware_pool3d_cuda"):
    return buildCUDAExtension(moduleName=name, buildDir=buildDir, rootDir=rootDir, 
                              sourceFiles=ROI_AWARE_POOL3D_SOURCES)

def buildPCDetModules(rootDir : str, buildDir : str):
    try:
        buildIou3dNmsModule(rootDir=rootDir, buildDir=buildDir)
        buildPointNet2Stack(rootDir=rootDir, buildDir=buildDir)
        buildRoiAwarePool3d(rootDir=rootDir, buildDir=buildDir)
        print('Built iou3d_nms_cuda, roiaware_pool3d_cuda, pointnet2_stack_cuda \n-------------')
    except:
        print("Couldn't build CUDA extensions")
        raise ImportError

# Add this part:
if __name__ == "__main__":
    # Get the current directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define build directory (you can modify this path as needed)
    build_dir = os.path.join(current_dir, "build")
    
    # Create build directory if it doesn't exist
    os.makedirs(build_dir, exist_ok=True)
    
    # Build the modules
    buildPCDetModules(rootDir=current_dir, buildDir=build_dir)      