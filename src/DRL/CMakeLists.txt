# DRL Framework CMake Configuration

# Find required packages
find_package(nlohmann_json REQUIRED)
find_package(onnxruntime REQUIRED)

# DRL library sources
set(DRL_SOURCES
    TelemetryData.cpp
    ModelMetadata.cpp
    EnvironmentAdapter.cpp
    ReplayBuffer.cpp
    DRLInference.cpp
    ActionDispatcher.cpp
    DRLDatabaseClient.cpp
    TelemetryStreamHandler.cpp
    DRLFramework.cpp
)

# Create DRL library
add_library(drl STATIC ${DRL_SOURCES})

# Include directories
target_include_directories(drl PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(drl PUBLIC
    nlohmann_json::nlohmann_json
    ${ONNXRUNTIME_LIBRARIES}
    pthread
)

# Set C++ standard
set_target_properties(drl PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Install targets
install(TARGETS drl
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
