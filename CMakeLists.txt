# ======================================================================== #
# Copyright 2021-2021 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()

project(CUDABitonic VERSION 0.1.1 LANGUAGES C CXX CUDA)

cmake_minimum_required(VERSION 3.16.2)
set (CMAKE_CXX_STANDARD 14)

# ==================================================================
# configure build type
# ==================================================================
if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  set(CUBIT_IS_SUBMODULE ON)
else()
  set(CUBIT_IS_SUBPROJECT OFF)
endif()
if (NOT CUBIT_IS_SUBPROJECT)
  if(NOT SET_UP_CONFIGURATIONS_DONE)
    set(SET_UP_CONFIGURATIONS_DONE 1)

    # No reason to set CMAKE_CONFIGURATION_TYPES if it's not a multiconfig generator
    # Also no reason mess with CMAKE_BUILD_TYPE if it's a multiconfig generator.
    if(CMAKE_CONFIGURATION_TYPES) # multiconfig generator?
      set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE) 
    else()
      if(NOT CMAKE_BUILD_TYPE)
	#            message("Defaulting to release build.")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
      endif()
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
      # set the valid options for cmake-gui drop-down list
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
    endif()
  endif()

  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()


set(CUDA_ARCHITECTURES "70;75;80;86")
#set(CUDA_ARCH_FLAGS "-gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86")
#set_source_files_properties(testing/test.cu PROPERTIES COMPILE_FLAGS ${CUDA_ARCH_FLAGS})
foreach(KEY_T int32 uint32 float uint64 double)
  add_executable(test_${KEY_T} testing/test.cu)
  target_compile_definitions(test_${KEY_T} PUBLIC -DKEY_T=${KEY_T})
  foreach(VALUE_T int32 uint32 float uint64 double)
    add_executable(test_${KEY_T}_${VALUE_T} testing/test.cu)
    target_compile_definitions(test_${KEY_T}_${VALUE_T} PUBLIC -DVALUE_T=${VALUE_T})
    target_compile_definitions(test_${KEY_T}_${VALUE_T} PUBLIC -DKEY_T=${KEY_T})
  endforeach()
endforeach()
