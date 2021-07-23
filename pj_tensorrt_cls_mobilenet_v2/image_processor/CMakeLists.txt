cmake_minimum_required(VERSION 3.0)

set(LibraryName "ImageProcessor")

# Create library
add_library (${LibraryName} image_processor.cpp image_processor.h classification_engine.cpp classification_engine.h)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LibraryName} ${OpenCV_LIBS})

set(INFERENCE_HELPER_DIR ${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper/)
# Link Common Helper module
add_subdirectory(${INFERENCE_HELPER_DIR}/CommonHelper CommonHelper)
target_include_directories(${LibraryName} PUBLIC ${INFERENCE_HELPER_DIR}/CommonHelper)
target_link_libraries(${LibraryName} CommonHelper)

# Link InferenceHelper module
set(INFERENCE_HELPER_ENABLE_OPENCV OFF CACHE BOOL "OPENCV" )
set(INFERENCE_HELPER_ENABLE_TENSORRT ON CACHE BOOL "TENSORRT")
add_subdirectory(${INFERENCE_HELPER_DIR}/InferenceHelper InferenceHelper)
target_include_directories(${LibraryName} PUBLIC ${INFERENCE_HELPER_DIR}/InferenceHelper)
target_link_libraries(${LibraryName} InferenceHelper)

