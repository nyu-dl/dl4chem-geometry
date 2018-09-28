add_library(rmsd SHARED
            rmsd.cpp
            theobald_rmsd.c
            center.c
            msvccompat.h
            sse_swizzle.h
            util.h
            )
target_include_directories(rmsd PUBLIC ${Tensorflow_INCLUDE_DIR})
target_compile_options(rmsd PRIVATE -msse2 -msse3 -msse4)
target_compile_options(rmsd PRIVATE -fopenmp)
target_link_libraries(rmsd PUBLIC gomp)
set_target_properties(rmsd PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      LIBRARY_OUTPUT_NAME rmsd.$<CONFIG>
                      )
