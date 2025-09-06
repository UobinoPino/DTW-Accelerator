cmake_minimum_required(VERSION 3.20)


# Find GTest package
find_package(GTest REQUIRED)

# Unit test executables
set(TEST_SOURCES
        test_core.cpp
        test_distance_metrics.cpp
        test_strategies.cpp
        test_constraints.cpp
)

# Create test executables
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})

    target_link_libraries(${test_name} PRIVATE
            dtw_accelerator
            GTest::GTest
            GTest::Main
    )

    # Add compile definitions based on available features
    if(USE_OPENMP)
        target_compile_definitions(${test_name} PRIVATE USE_OPENMP)
    endif()

    if(USE_MPI)
        target_compile_definitions(${test_name} PRIVATE USE_MPI)
    endif()

    if(USE_CUDA)
        target_compile_definitions(${test_name} PRIVATE USE_CUDA)
    endif()

    # Register test with CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Add custom target to run all unit tests
add_custom_target(run_unit_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS ${TEST_SOURCES}
        COMMENT "Running unit tests"
)

# Add performance subdirectory if benchmarks are enabled
if(BUILD_BENCHMARKS)
    add_subdirectory(performance)
endif()

# Install test executables (optional)
install(TARGETS test_core test_distance_metrics test_strategies test_constraints
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/tests
)