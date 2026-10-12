#!/bin/bash
# 本机构建内核所需的主机工具环境（免 root）
#
# 背景：本机 Ubuntu 24.04 没有 flex / bison，且 sudo 需要密码（不能 apt install）。
# 方案：用 `apt-get download` 取 deb（普通用户即可），`dpkg-deb -x` 解到 ~/.local/kbtools/root，
#       再把 bin 挂进 PATH。两个坑：
#         1) bison 运行时需要 m4，但 Debian 把它编译成写死的 /usr/bin/m4 → 用 M4 环境变量覆盖
#         2) bison 的 skeleton 在 /usr/share/bison → 用 BISON_PKGDATADIR 覆盖
#
# 用法：source scripts/build/host-tools-env.sh
#
# 需要的 deb（已下载在 ~/.local/kbtools/debs/）：
#   flex_2.6.4-8.2build1_amd64.deb                    kconfig / dtc 词法分析
#   bison_2:3.8.2+dfsg-1ubuntu0.24.04.1_amd64.deb     kconfig / dtc 语法分析
#   m4_1.4.19-4build1_amd64.deb                       bison 的运行时依赖
#   libsigsegv2_2.14-1ubuntu2_amd64.deb               m4 的依赖
#   u-boot-tools_2025.10-0ubuntu0.24.04.3_amd64.deb   mkimage：dtb overlay 的 fixup.scr
#   device-tree-compiler_1.7.0-2build1_amd64.deb      dtc / fdt*：设备树检查
#   libssl-dev_3.0.13-0ubuntu3.16_amd64.deb           ★ certs/extract-cert 要 <openssl/bio.h>
#                                                     （CONFIG_SYSTEM_TRUSTED_KEYRING=y 就会编它，
#                                                       跟 MODULE_SIG 开不开无关 —— 这点我判断错过一次）
#
# 一键补齐（普通用户即可，不需要 root）：scripts/build/setup-host-tools.sh
#
# ★ 架构陷阱：arm64 的 u-boot-tools 与 amd64 **同名**。两者混在一个目录里解包，
#   aarch64 的 mkimage 会覆盖主机版本，构建会在毫无提示的情况下失败。
#   → amd64/all 放 ~/.local/kbtools/debs/（解包到 root/）；arm64 放 ~/.local/kbtools/board-debs/（只送板子）。

KBT="$HOME/.local/kbtools"
_missing=""
for t in flex bison m4 mkimage; do
  [ -x "$KBT/root/usr/bin/$t" ] || _missing="$_missing $t"
done
if [ -n "$_missing" ]; then
  echo "✗ 主机工具未就绪（缺:$_missing）：跑 scripts/build/setup-host-tools.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="$KBT/root/usr/bin:$PATH"
export M4="$KBT/root/usr/bin/m4"
export BISON_PKGDATADIR="$KBT/root/usr/share/bison"
# libcrypto 的 .pc 在本会话里刚下、还没进系统默认搜索路径，必须显式指过去
export PKG_CONFIG_PATH="$KBT/root/usr/lib/x86_64-linux-gnu/pkgconfig:$KBT/root/usr/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

# ARM 官方工具链（交叉编译 aarch64）
RKROOT="${RKROOT:-$HOME/rk-kernel}"
if [ -d "$RKROOT/toolchain/arm-gnu-toolchain/bin" ]; then
  export PATH="$RKROOT/toolchain/arm-gnu-toolchain/bin:$PATH"
fi
export ARCH=arm64
export CROSS_COMPILE=aarch64-none-linux-gnu-
