#!/usr/bin/env bash
# Rocky Linux 10 构建依赖的安装与自检。用法：install_deps_rocky10.sh [check|install]
#
# 依赖来源：ninfer-4090/CMakeLists.txt 与同仓 Dockerfile，而不是猜测。
#   - FFmpeg 开发包：pkg_check_modules(FFMPEG REQUIRED ...) 位于非 WIN32 分支顶层，
#     不受任何 option 门控制，即使 -DNINFER_BUILD_APPS=OFF 也强制要求。
#   - libcurl：仅在 NINFER_BUILD_APPS=ON 或 BUILD_TESTING=ON 时启用（媒体采集模块）。
#   - xgrammar：由 CMake FetchContent 从 GitHub 拉取，不是系统包，本脚本不处理。
#   - cpp-httplib / nlohmann / utf8proc：已随源码树内置于 third_party/，无需安装。
set -euo pipefail

mode="${1:-check}"

# 四件套必须同版本族，且需满足 CMakeLists 中的下限：avformat>=60 avcodec>=60 avutil>=58 swscale>=7
pkgs_ffmpeg=(libavformat-free-devel libavcodec-free-devel libavutil-free-devel libswscale-free-devel)
pkgs_tools=(cmake ninja-build pkgconf-pkg-config)
pkgs_curl=(libcurl-devel)
# just 只服务仓根的 justfile —— 它是脚本的"目录"，不是构建依赖。缺了它脚本照样能跑，但这个仓
# 把 just 当作对外入口，所以这里按必需项检查。
pkgs_just=(just)

# pkg-config 模块名 -> 版本下限，取自 CMakeLists.txt
pc_specs=("libavformat:60" "libavcodec:60" "libavutil:58" "libswscale:7" "libcurl:7.85")

verify() {
  local rc=0
  echo "--- 工具链"
  local tool
  for tool in cmake ninja pkg-config nvcc g++ just; do
    if command -v "${tool}" >/dev/null 2>&1; then
      echo "  存在  ${tool}  $(command -v "${tool}")"
    else
      echo "  缺失  ${tool}"
      rc=1
    fi
  done

  echo "--- pkg-config 模块（含 CMakeLists 的版本下限）"
  local spec name need got
  for spec in "${pc_specs[@]}"; do
    name="${spec%%:*}"
    need="${spec##*:}"
    if ! pkg-config --exists "${name}" 2>/dev/null; then
      echo "  缺失  ${name} (需 >= ${need})"
      rc=1
      continue
    fi
    got="$(pkg-config --modversion "${name}")"
    if pkg-config --atleast-version="${need}" "${name}"; then
      echo "  满足  ${name} ${got} (需 >= ${need})"
    else
      echo "  过低  ${name} ${got} (需 >= ${need})"
      rc=1
    fi
  done

  echo "--- 头文件可达性（-I 取自 pkg-config）"
  local scratch="${NINFER_TERNARY_TMPDIR:-${TMPDIR:-/tmp}/ninfer-ternary}"
  mkdir -p "${scratch}"
  local probe="${scratch}/dep_probe_$$.cc"
  printf '#include <libavformat/avformat.h>\n#include <libavcodec/avcodec.h>\n#include <libavutil/imgutils.h>\n#include <libswscale/swscale.h>\n#include <curl/curl.h>\nint main(){return 0;}\n' > "${probe}"
  if g++ -std=c++20 -fsyntax-only $(pkg-config --cflags libavformat libavcodec libavutil libswscale libcurl 2>/dev/null) "${probe}" 2>/dev/null; then
    echo "  通过  ffmpeg + curl 头文件可编译"
  else
    echo "  失败  ffmpeg 或 curl 头文件不可达"
    rc=1
  fi
  rm -f "${probe}"

  return "${rc}"
}

if [[ "${mode}" == "check" ]]; then
  verify
  exit $?
fi

if [[ "${mode}" != "install" ]]; then
  echo "用法：$(basename "$0") [check|install]" >&2
  exit 2
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "install 模式需要 root（dnf 写系统目录）" >&2
  exit 3
fi

# EPEL 提供 ffmpeg-free 系列；Rocky 10 默认不含 FFmpeg 开发包
if ! rpm -q epel-release >/dev/null 2>&1; then
  echo "=== 启用 EPEL ==="
  dnf install -y epel-release
fi

# install_weak_deps=False 让事务从 241 个包降到约 98 个：弱依赖会拖入 pipewire、
# samba、tesseract 等与解码无关的桌面栈
echo "=== 安装构建依赖 ==="
dnf install -y --setopt=install_weak_deps=False \
  "${pkgs_tools[@]}" "${pkgs_ffmpeg[@]}" "${pkgs_curl[@]}" "${pkgs_just[@]}"

echo "=== 安装后自检 ==="
verify

echo "=== cmake 版本（CMakeLists 要求 >= 3.28）==="
cmake --version | head -1

echo "=== 后续步骤 ==="
echo "  export NINFER_ROOT=<ninfer-4090 源码树>"
echo "  $(dirname "$0")/build.sh incremental -- -DNINFER_BUILD_APPS=ON"
echo "  xgrammar 需在 configure 阶段联网从 GitHub 拉取（git clone mlc-ai/xgrammar v0.2.5.post1）"