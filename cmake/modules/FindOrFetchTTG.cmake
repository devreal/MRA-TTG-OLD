if (NOT TARGET ttg)
  find_package(ttg CONFIG)
endif(NOT TARGET ttg)

if (TARGET ttg)
    message(STATUS "Found ttg CONFIG at ${ttg_CONFIG}")
else (TARGET ttg)

  include(FetchContent)
#  FetchContent_Declare(
#      ttg
#      GIT_REPOSITORY      https://github.com/devreal/ttg.git
#      GIT_TAG             6061bc5ad5172b82279a7dc090afa5f391a3fdce
#  )

FetchContent_Declare(
      ttg
      GIT_REPOSITORY      https://github.com/TESSEorg/ttg.git
      GIT_TAG             b180cac033475a7b9663514035e0f7f62decddf1
  )

  FetchContent_MakeAvailable(ttg)
  FetchContent_GetProperties(ttg
      SOURCE_DIR TTG_SOURCE_DIR
      BINARY_DIR TTG_BINARY_DIR
      )

endif(TARGET ttg)

# postcond check
if (NOT TARGET ttg)
message(FATAL_ERROR "FindOrFetchTTG could not make ttg target available")
endif(NOT TARGET ttg)
