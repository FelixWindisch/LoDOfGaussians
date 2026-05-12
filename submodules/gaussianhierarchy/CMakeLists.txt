cmake_minimum_required (VERSION 3.0)
project (GaussianHierarchy LANGUAGES CUDA CXX)

add_library (GaussianHierarchy
 FlatGenerator.h
 FlatGenerator.cpp
 PointbasedKdTreeGenerator.h
 PointbasedKdTreeGenerator.cpp
 ClusterMerger.h
 ClusterMerger.cpp
 appearance_filter.h
 appearance_filter.cpp
 AvgMerger.h
 AvgMerger.cpp
 writer.h
 writer.cpp
 common.h 
 loader.h 
 loader.cpp
 hierarchy_loader.h 
 hierarchy_loader.cpp
 hierarchy_explicit_loader.h
 hierarchy_explicit_loader.cpp
 hierarchy_writer.h
 hierarchy_writer.cpp
 traversal.h
 traversal.cpp
 runtime_maintenance.h
 runtime_maintenance.cu
 runtime_switching.h
 runtime_switching.cu
 utils.cu
 utils.h
 rotation_aligner.h
 rotation_aligner.cpp
 half.hpp
 types.h)
target_include_directories(GaussianHierarchy PRIVATE dependencies/eigen)
target_include_directories(GaussianHierarchy PRIVATE dependencies/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_property(TARGET GaussianHierarchy PROPERTY CXX_STANDARD 17)
set_target_properties(GaussianHierarchy PROPERTIES CUDA_ARCHITECTURES "70;75;86")
set(CMAKE_CUDA_STANDARD 17)

target_include_directories(GaussianHierarchy
                           PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )
						   
install(TARGETS
        GaussianHierarchy
    EXPORT GaussianHierarchyTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES runtime_maintenance.h runtime_switching.h hierarchy_loader.h types.h utils.h DESTINATION include)
install(EXPORT GaussianHierarchyTargets
  FILE GaussianHierarchyConfig.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

add_executable (GaussianHierarchyCreator
 mainHierarchyCreator.cpp
)

add_executable (GaussianHierarchyMerger
 mainHierarchyMerger.cpp
)

target_include_directories(GaussianHierarchyCreator PRIVATE dependencies/eigen)
target_include_directories(GaussianHierarchyCreator PRIVATE dependencies/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_property(TARGET GaussianHierarchyCreator PROPERTY CXX_STANDARD 17)
target_link_libraries(GaussianHierarchyCreator PUBLIC GaussianHierarchy)

target_include_directories(GaussianHierarchyMerger PRIVATE dependencies/eigen)
target_include_directories(GaussianHierarchyMerger PRIVATE dependencies/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_property(TARGET GaussianHierarchyMerger PROPERTY CXX_STANDARD 17)
target_link_libraries(GaussianHierarchyMerger PUBLIC GaussianHierarchy)