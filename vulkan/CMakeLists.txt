set(API vulkan)
set(TARGET_NAME KullaContyIBL-Vulkan)
file(
        GLOB_RECURSE SRCS
        "${PROJECT_SOURCE_DIR}/${API}/*.cpp"
        "${PROJECT_SOURCE_DIR}/${API}/*.hpp"
)

add_executable(${TARGET_NAME} ${SRCS})

target_link_libraries(${TARGET_NAME} PRIVATE CGUtils)

target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/ext/cy)

target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)

target_include_directories(${TARGET_NAME} PRIVATE
        ../ext/vulkan-headers/include)

target_compile_definitions(${TARGET_NAME} PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

target_link_libraries(
        ${TARGET_NAME} PRIVATE
        glslang
        glslang-default-resource-limits
        SPIRV
)
