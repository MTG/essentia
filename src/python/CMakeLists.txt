# Python bindings CMakeLists.txt

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find NumPy
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT NUMPY_INCLUDE_DIR)
    message(FATAL_ERROR "NumPy is required for Python bindings but was not found")
endif()

# Find pybind11 (required for Python bindings)
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Download and build pybind11 if not found
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.4
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Get Python site packages directory
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Collect Python binding sources
file(GLOB PYTHON_BINDING_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/sonoria.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/parsing.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pytypes/*.cpp"
)

# Create Python extension module
pybind11_add_module(_sonoria ${PYTHON_BINDING_SOURCES})

target_include_directories(_sonoria PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/pytypes"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/sonoria"
)

target_link_libraries(_sonoria PRIVATE sonoria Python3::Module)

# Set output name and installation path
set_target_properties(_sonoria PROPERTIES
    OUTPUT_NAME "_sonoria"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/sonoria"
)

# Install Python module
install(TARGETS _sonoria
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}/sonoria"
)

# Install Python scripts
file(GLOB_RECURSE PYTHON_SCRIPTS
    "${CMAKE_CURRENT_SOURCE_DIR}/sonoria/*.py"
)

foreach(script ${PYTHON_SCRIPTS})
    file(RELATIVE_PATH rel_path "${CMAKE_CURRENT_SOURCE_DIR}/sonoria" ${script})
    get_filename_component(script_dir ${rel_path} DIRECTORY)
    install(FILES ${script}
        DESTINATION "${PYTHON_SITE_PACKAGES}/sonoria/${script_dir}"
    )
endforeach()

message(STATUS "Python site packages: ${PYTHON_SITE_PACKAGES}")
message(STATUS "NumPy include dir: ${NUMPY_INCLUDE_DIR}")
