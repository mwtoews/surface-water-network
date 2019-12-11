cmake_minimum_required(VERSION 3.7)
project(MODFLOW-2005 LANGUAGES Fortran C)

file(GLOB SOURCE src/*.f* src/*.c)
add_executable(mf2005 ${SOURCE})

include(WriteOpenspecInc)
write_openspec_inc(src/openspec.inc mf2005)

set(C_WARN_FLAGS
    -Wall
    -Wextra
)
set(Fortran_WARN_FLAGS
    -Wall
    -Wextra
    -Wno-tabs
    -Wno-compare-reals
    -Wno-conversion
    -Wno-unused-dummy-argument
    -Wno-unused-variable
    -Wno-unused-label
    -Wno-unused-parameter
    -Wno-maybe-uninitialized
)
set(Fortran_BUILD_FLAGS
    -ffpe-summary=overflow
    -ffpe-trap=overflow,zero,invalid
)
target_compile_options(mf2005
  PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${Fortran_WARN_FLAGS}>
  PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${Fortran_BUILD_FLAGS}>
  PRIVATE $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
)

install(TARGETS mf2005
  RUNTIME DESTINATION bin
)
