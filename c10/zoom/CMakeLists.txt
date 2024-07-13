include(../../cmake/public/utils.cmake)

# ---[ Configure macro file.
set(C10_ZOOM_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
# configure_file(
#     ${CMAKE_CURRENT_LIST_DIR}/impl/hip_cmake_macros.h.in
#     ${CMAKE_BINARY_DIR}/c10/hip/impl/hip_cmake_macros.h)

# NB: All previous cu files are renamed into cc files.  This isn't tested at the
# moment.
file(GLOB C10_ZOOM_SRCS
        *.cpp
        *.cu
        impl/*.cpp
        impl/*.cu
        )

# Mark the cc files as HIP files, so we call the compiler.  (They have to be
# suffixed with cc, because the hcc compiler won't accept them otherwise.)
file(GLOB __c10_zoom_srcs_cpp *.cu impl/*.cu)
set_source_files_properties(${__c10_zoom_srcs_cpp} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

file(GLOB_RECURSE C10_HIP_HEADERS *.h)
hip_add_library(c10_zoom ${C10_HIP_SRCS} ${C10_HIP_HEADERS})

# Propagate HIP_CXX_FLAGS that were set from Dependencies.cmake
target_compile_options(c10_zoom PRIVATE ${HIP_CXX_FLAGS})

# caffe2_hip adds a bunch of dependencies like rocsparse, but c10/hip is supposed to be
# minimal.  I'm not sure if we need hip_hcc or not; for now leave it out

# If building shared library, set dllimport/dllexport proper.
target_compile_options(c10_zoom PRIVATE "-DC10_ZOOM_BUILD_MAIN_LIB")
# Enable hidden visibility if compiler supports it.
if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(c10_zoom PRIVATE "-fvisibility=hidden")
endif()

# ---[ Dependency of c10_zoom
target_link_libraries(c10_zoom PUBLIC c10)

target_link_libraries(c10_zoom PUBLIC ${PYTORCH_HIP_LIBRARIES})

target_include_directories(
    c10_zoom PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

# add_subdirectory(test)

# ---[ Installation
install(TARGETS c10_zoom EXPORT Caffe2Targets DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
# install(FILES ${CMAKE_BINARY_DIR}/c10/hip/impl/hip_cmake_macros.h
#   DESTINATION include/c10/hip/impl)
