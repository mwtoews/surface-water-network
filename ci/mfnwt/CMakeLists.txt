cmake_minimum_required(VERSION 3.7)
project(MODFLOW-NWT LANGUAGES Fortran)

file(GLOB SOURCE src/*.f*)
add_executable(mfnwt ${SOURCE})

include(WriteOpenspecInc)
write_openspec_inc(src/openspec.inc mfnwt)

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
target_compile_options(mfnwt
  PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${Fortran_WARN_FLAGS}>
  PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${Fortran_BUILD_FLAGS}>
)

install(TARGETS mfnwt
  RUNTIME DESTINATION bin
)
