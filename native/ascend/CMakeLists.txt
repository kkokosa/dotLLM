cmake_minimum_required(VERSION 3.18)
project(dotllm_ascend VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "The Ascend bridge requires 64-bit Linux.")
endif()

set(ASCEND_TOOLKIT_ROOT "$ENV{ASCEND_TOOLKIT_HOME}" CACHE PATH "CANN toolkit root")
if(NOT ASCEND_TOOLKIT_ROOT)
    set(ASCEND_TOOLKIT_ROOT "/usr/local/Ascend/ascend-toolkit/latest")
endif()
find_path(CANN_INCLUDE_DIR acl/acl.h
    HINTS "${ASCEND_TOOLKIT_ROOT}/include" REQUIRED)
find_path(CANN_OP_INCLUDE_DIR aclnnop/aclnn_matmul.h
    HINTS "${ASCEND_TOOLKIT_ROOT}/include" REQUIRED)
foreach(library ascendcl nnopbase opapi)
    find_library(CANN_${library}_LIBRARY NAMES ${library}
        HINTS "${ASCEND_TOOLKIT_ROOT}/lib64" REQUIRED)
    list(APPEND CANN_LIBRARIES "${CANN_${library}_LIBRARY}")
endforeach()

add_library(dotllm_ascend SHARED src/ops.cpp)
target_compile_features(dotllm_ascend PRIVATE cxx_std_17)
target_include_directories(dotllm_ascend PRIVATE include "${CANN_INCLUDE_DIR}" "${CANN_OP_INCLUDE_DIR}")
target_link_libraries(dotllm_ascend PRIVATE ${CANN_LIBRARIES})
target_compile_options(dotllm_ascend PRIVATE -Wall -Wextra -Werror)
target_link_options(dotllm_ascend PRIVATE -Wl,--no-undefined)
set_target_properties(dotllm_ascend PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
    INSTALL_RPATH "$ORIGIN")
install(TARGETS dotllm_ascend LIBRARY DESTINATION lib)
install(FILES include/dotllm_ascend.h DESTINATION include)
