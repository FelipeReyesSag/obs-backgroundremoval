# SPDX-FileCopyrightText: 2026 Felipe Reyes
#
# SPDX-License-Identifier: GPL-3.0-or-later

# OnnxRuntimeHelper.cmake
#
# Locates a prebuilt ONNX Runtime distribution downloaded from Microsoft's
# GitHub releases (https://github.com/microsoft/onnxruntime/releases) and
# exposes it as the imported target `onnxruntime::onnxruntime`.
#
# The prebuilt tarballs / zips all share the same layout:
#
#   <prefix>/
#   ├── include/
#   │   ├── onnxruntime_cxx_api.h
#   │   ├── cpu_provider_factory.h
#   │   ├── coreml_provider_factory.h         (macOS universal/arm64 only)
#   │   ├── dml_provider_factory.h            (Windows DirectML variant only)
#   │   └── ...
#   └── lib/
#       ├── libonnxruntime.<version>.dylib    (macOS)
#       ├── libonnxruntime.so.<version>       (Linux)
#       └── onnxruntime.dll / onnxruntime.lib (Windows)
#
# Callers should set ONNXRUNTIME_ROOT to the extracted directory (either as a
# cache variable passed on the cmake command line, or via the environment).
# If unset, the helper falls back to `${CMAKE_SOURCE_DIR}/.deps_vendor/onnxruntime`.
#
# After a successful setup the helper also defines the following variables in
# the parent scope (they are only used to drive optional EP compile flags in
# the plugin target):
#   HAVE_ONNXRUNTIME_COREML_EP - TRUE if coreml_provider_factory.h is present
#   HAVE_ONNXRUNTIME_DML_EP    - TRUE if dml_provider_factory.h    is present
#   HAVE_ONNXRUNTIME_CUDA_EP   - TRUE if cuda_provider_factory.h   is present
#
# Also sets ONNXRUNTIME_RUNTIME_LIBS to the list of runtime artifacts that
# must be bundled next to the plugin (the .dylib / .so.X / .dll / direct
# dependencies such as DirectML.dll).

include_guard(GLOBAL)

function(obs_plate_blur_setup_onnxruntime)
  # 1. Resolve the prefix directory.
  if(NOT ONNXRUNTIME_ROOT)
    if(DEFINED ENV{ONNXRUNTIME_ROOT} AND NOT "$ENV{ONNXRUNTIME_ROOT}" STREQUAL "")
      set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    else()
      set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/.deps_vendor/onnxruntime")
    endif()
  endif()

  if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(
      FATAL_ERROR
      "Could not find onnxruntime_cxx_api.h under ${ONNXRUNTIME_ROOT}/include. "
      "Download a prebuilt release from https://github.com/microsoft/onnxruntime/releases, "
      "extract it, and point ONNXRUNTIME_ROOT at the extracted directory."
    )
  endif()

  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
  set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

  # 2. Find the primary library file.
  set(_ort_runtime_libs "")
  if(WIN32)
    set(_ort_import_lib "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
    set(_ort_runtime_lib "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll")
    if(NOT EXISTS "${_ort_import_lib}")
      message(FATAL_ERROR "Missing ${_ort_import_lib}")
    endif()
    if(NOT EXISTS "${_ort_runtime_lib}")
      message(FATAL_ERROR "Missing ${_ort_runtime_lib}")
    endif()

    add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(
      onnxruntime::onnxruntime
      PROPERTIES
        IMPORTED_LOCATION "${_ort_runtime_lib}"
        IMPORTED_IMPLIB "${_ort_import_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )
    list(APPEND _ort_runtime_libs "${_ort_runtime_lib}")

    # Some prebuilt Windows zips ship DirectML.dll alongside onnxruntime.dll.
    if(EXISTS "${ONNXRUNTIME_LIB_DIR}/DirectML.dll")
      list(APPEND _ort_runtime_libs "${ONNXRUNTIME_LIB_DIR}/DirectML.dll")
    endif()

    # And provider DLLs if present (CUDA / TRT variants).
    file(GLOB _ort_provider_dlls "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_*.dll")
    list(APPEND _ort_runtime_libs ${_ort_provider_dlls})
  elseif(APPLE)
    file(GLOB _ort_dylibs "${ONNXRUNTIME_LIB_DIR}/libonnxruntime*.dylib")
    if(NOT _ort_dylibs)
      message(FATAL_ERROR "No libonnxruntime*.dylib found under ${ONNXRUNTIME_LIB_DIR}")
    endif()

    # Prefer the versioned file (libonnxruntime.<version>.dylib) as the real
    # library; the unversioned libonnxruntime.dylib is usually a symlink to it.
    set(_ort_primary_lib "")
    foreach(_cand IN LISTS _ort_dylibs)
      if(NOT IS_SYMLINK "${_cand}")
        set(_ort_primary_lib "${_cand}")
        break()
      endif()
    endforeach()
    if(NOT _ort_primary_lib)
      list(GET _ort_dylibs 0 _ort_primary_lib)
    endif()

    add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(
      onnxruntime::onnxruntime
      PROPERTIES IMPORTED_LOCATION "${_ort_primary_lib}" INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )
    # Ship every file we saw - symlinks included - so that @rpath/libonnxruntime.dylib
    # inside the linked plugin resolves at runtime.
    list(APPEND _ort_runtime_libs ${_ort_dylibs})
  else()
    file(GLOB _ort_sos "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so*")
    if(NOT _ort_sos)
      message(FATAL_ERROR "No libonnxruntime.so* found under ${ONNXRUNTIME_LIB_DIR}")
    endif()

    # Same logic - prefer the concrete file, not the SONAME symlink.
    set(_ort_primary_lib "")
    foreach(_cand IN LISTS _ort_sos)
      if(NOT IS_SYMLINK "${_cand}")
        set(_ort_primary_lib "${_cand}")
        break()
      endif()
    endforeach()
    if(NOT _ort_primary_lib)
      list(GET _ort_sos 0 _ort_primary_lib)
    endif()

    add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(
      onnxruntime::onnxruntime
      PROPERTIES IMPORTED_LOCATION "${_ort_primary_lib}" INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )
    list(APPEND _ort_runtime_libs ${_ort_sos})
  endif()

  # 3. Probe for execution-provider headers so CMakeLists can conditionally
  # add -DHAVE_ONNXRUNTIME_*_EP to the plugin target.
  set(_have_coreml FALSE)
  set(_have_dml FALSE)
  set(_have_cuda FALSE)
  if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/coreml_provider_factory.h")
    set(_have_coreml TRUE)
  endif()
  if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/dml_provider_factory.h")
    set(_have_dml TRUE)
  endif()
  if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/cuda_provider_factory.h")
    set(_have_cuda TRUE)
  endif()

  set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" PARENT_SCOPE)
  set(ONNXRUNTIME_RUNTIME_LIBS "${_ort_runtime_libs}" PARENT_SCOPE)
  set(HAVE_ONNXRUNTIME_COREML_EP ${_have_coreml} PARENT_SCOPE)
  set(HAVE_ONNXRUNTIME_DML_EP ${_have_dml} PARENT_SCOPE)
  set(HAVE_ONNXRUNTIME_CUDA_EP ${_have_cuda} PARENT_SCOPE)

  message(STATUS "ONNX Runtime prefix : ${ONNXRUNTIME_ROOT}")
  message(STATUS "  CoreML provider   : ${_have_coreml}")
  message(STATUS "  DirectML provider : ${_have_dml}")
  message(STATUS "  CUDA provider     : ${_have_cuda}")
endfunction()
