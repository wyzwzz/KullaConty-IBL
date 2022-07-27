set(API opengl)
set(TARGET_NAME KullaContyIBL-OpenGL)
file(
        GLOB_RECURSE SRCS
        "${PROJECT_SOURCE_DIR}/${API}/*.cpp"
        "${PROJECT_SOURCE_DIR}/${API}/*.hpp"
)



add_executable(${TARGET_NAME} ${SRCS})

target_link_libraries(${TARGET_NAME} PRIVATE CGUtils)

target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/ext/cy)

target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)