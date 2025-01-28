include_directories(madness/include)

add_executable(mra-device-host-parsec
               madness/src/mra-ttg-device.cc
               madness/src/gl.cc
               madness/src/twoscale.cc)
target_compile_definitions(mra-device-host-parsec PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-device-host-parsec PUBLIC MADworld ttg-parsec)

add_executable(mra-gaxpytest madness/tests/gaxpy_test.cc)
target_compile_definitions(mra-gaxpytest PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-gaxpytest PUBLIC MADworld ttg-parsec)

add_executable(mra-tensortest madness/tests/tensor_test.cc)
target_compile_definitions(mra-tensortest PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-tensortest PUBLIC MADworld ttg-parsec)

if (HAVE_CUDA)
  add_executable(mra-device-cuda-parsec
                     madness/src/mra-ttg-device.cu
                     madness/src/gl.cc
                     madness/src/twoscale.cc)
  target_compile_definitions(mra-device-cuda-parsec PUBLIC MRA_ENABLE_CUDA=1 TTG_USE_PARSEC=1)
  target_compile_options(mra-device-cuda-parsec PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(mra-device-cuda-parsec PUBLIC MADworld ttg-parsec)
endif (HAVE_CUDA)

if (HAVE_HIP)
  add_executable(mra-device-hip-parsec
                     madness/src/mra-ttg-device.hip
                     madness/src/gl.cc
                     madness/src/twoscale.cc)
  target_compile_definitions(mra-device-hip-parsec PUBLIC MRA_ENABLE_HIP=1 TTG_USE_PARSEC=1)
  target_link_libraries(mra-device-hip-parsec PUBLIC MADworld ttg-parsec hip::host)
endif (HAVE_HIP)

# test to verify that tensor slices work
add_executable(tensor-test
               madness/tests/tensor_test.cc)
target_compile_definitions(tensor-test PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(tensor-test PUBLIC MADworld ttg-parsec)

