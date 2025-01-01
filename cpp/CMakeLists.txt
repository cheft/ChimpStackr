cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0167 NEW)

# Set the project name
project(MyFocusStack)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Debug)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# 查找Boost库
find_package(Boost REQUIRED)

# 包含Boost目录
include_directories(${Boost_INCLUDE_DIRS})

# Include NumCpp
include_directories(/usr/local/include) # Adjust this path

# Add the executable
add_executable(MyFocusStack main.cpp Algorithm.cpp)

# Link the OpenCV libraries
target_link_libraries(MyFocusStack ${OpenCV_LIBS})

# If using NumCpp, link any required libraries
# target_link_libraries(MyFocusStack numcpp) # Uncomment this if NumCpp requires linking

# Set the output directory for the executable
set_target_properties(MyFocusStack PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

target_compile_options(MyFocusStack PRIVATE -g)
