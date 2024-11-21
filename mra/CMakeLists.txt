include_directories(madness/include)

add_executable(mra-device-host-parsec
               madness/src/mra-ttg-device.cc
               madness/src/gl.cc
               madness/src/twoscale.cc)
target_compile_definitions(mra-device-host-parsec PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-device-host-parsec PUBLIC MADworld ttg-parsec)

if (HAVE_CUDA)
  add_executable(mra-device-cuda-parsec
                     madness/src/mra-ttg-device.cu
                     madness/src/gl.cc
                     madness/src/twoscale.cc)
  target_compile_definitions(mra-device-cuda-parsec PUBLIC TTG_ENABLE_CUDA=1 TTG_USE_PARSEC=1)
  target_compile_options(mra-device-cuda-parsec PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(mra-device-cuda-parsec PUBLIC MADworld ttg-parsec)

endif (HAVE_CUDA)