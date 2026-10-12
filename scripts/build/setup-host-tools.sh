#!/bin/bash
# 一键补齐内核构建所需的主机工具（免 root）
#
# 为什么需要这个脚本：这台构建机（Ubuntu 24.04）**没有** flex / bison / m4 / mkimage / dtc，
# 而 sudo 需要密码、本会话审批关闭 → 装不了包。
# 办法：用 `apt-get download`（普通用户就能用）取 deb，再用 `dpkg-deb -x` 解到
#       ~/.local/kbtools/root，把 bin 挂进 PATH。全程不需要 root。
#
# ★ 架构陷阱（差点静默坏事）：
#   arm64 的 u-boot-tools 与 amd64 的**同名**，如果混在同一个目录里一起解包，
#   aarch64 的 mkimage 会覆盖掉主机用的 x86-64 版本 —— 构建会在毫无提示的情况下失败。
#   所以这里把两者物理隔离：
#     $KBT/debs/       放 amd64/all 的 deb（解到 $KBT/root，供本机构建用）
#     $KBT/board-debs/ 放 arm64 的 deb（只打包送给板子，绝不在此解包）
#
# 用法：scripts/build/setup-host-tools.sh
set -euo pipefail

KBT="${KBT:-$HOME/.local/kbtools}"
DEBS="$KBT/debs"
BDEBS="$KBT/board-debs"
ROOT="$KBT/root"

# 本机构建需要的 amd64 包
HOST_PKGS="flex bison m4 libsigsegv2 u-boot-tools device-tree-compiler libssl-dev"
# 只送给板子、不在本机解包的 arm64 包
BOARD_PKGS="u-boot-tools"

echo "═══ 补齐主机构建工具（免 root）═══"
mkdir -p "$DEBS" "$BDEBS"

echo "▶ 下载 amd64 包: $HOST_PKGS"
cd "$DEBS"
for p in $HOST_PKGS; do
  if ls "${p}"_*.deb >/dev/null 2>&1; then
    echo "  已有 $p"
  else
    apt-get download "$p" >/dev/null 2>&1 && echo "  下载 $p" || echo "  ⚠ $p 下载失败（可能已装或在源里没有）"
  fi
done

echo "▶ 下载 arm64 包（板子兜底用）: $BOARD_PKGS"
cd "$BDEBS"
for p in $BOARD_PKGS; do
  if ls "${p}"_*arm64.deb >/dev/null 2>&1; then
    echo "  已有 $p:arm64"
  else
    # 注意：默认源是 amd64，arm64 要从 ports.ubuntu.com 直接拉
    ver="$(apt-cache policy "$p" 2>/dev/null | awk '/Candidate:/{print $2}')"
    if [ -n "$ver" ]; then
      url="http://ports.ubuntu.com/ubuntu-ports/pool/main/u/u-boot/${p}_${ver}_arm64.deb"
      curl -sfL -o "${p}_${ver}_arm64.deb" "$url" && echo "  下载 $p:arm64 ($ver)" \
        || { rm -f "${p}_${ver}_arm64.deb"; echo "  ⚠ $p:arm64 下载失败"; }
    else
      echo "  ⚠ 查不到 $p 的候选版本"
    fi
  fi
done

echo "▶ 解包 amd64 包 → $ROOT"
mkdir -p "$ROOT"
# 只解 amd64/all，绝不碰 arm64 —— 见头部"架构陷阱"
n=0
for d in "$DEBS"/*.deb; do
  [ -e "$d" ] || continue
  case "$d" in
    *_arm64.deb|*_armhf.deb) echo "  跳过（非本机架构）: ${d##*/}"; continue ;;
  esac
  dpkg-deb -x "$d" "$ROOT"
  n=$((n + 1))
done
echo "  解包 $n 个"

echo "▶ 修正 pkg-config 的 prefix（deb 里的 .pc 写死了 /usr，我们装在别处）"
fixed=0
for pc in "$ROOT"/usr/lib/x86_64-linux-gnu/pkgconfig/*.pc "$ROOT"/usr/share/pkgconfig/*.pc; do
  [ -e "$pc" ] || continue
  if grep -q '^prefix=/usr$' "$pc"; then
    sed -i "s|^prefix=/usr$|prefix=$ROOT/usr|" "$pc"
    fixed=$((fixed + 1))
  fi
done
echo "  修正 $fixed 个 .pc"

# 第二个 openssl 坑：openssl/opensslconf.h 不在 /usr/include/openssl 下，而在
# **multiarch 目录** /usr/include/x86_64-linux-gnu/openssl/。真机上 gcc 默认就搜这个路径，
# 所以 .pc 的 `Cflags: -I${includedir}` 够用；我们解到别处后这个默认路径没了，
# 于是 bio.h 能编到、它 include 的 opensslconf.h 却找不到。
# → 往 .pc 的 Cflags 里显式补上 multiarch 头目录。
MULTIARCH_INC="$ROOT/usr/include/x86_64-linux-gnu"
if [ -d "$MULTIARCH_INC" ]; then
  echo "▶ 补 multiarch 头目录到 .pc 的 Cflags: $MULTIARCH_INC"
  for pc in "$ROOT"/usr/lib/x86_64-linux-gnu/pkgconfig/*.pc; do
    [ -e "$pc" ] || continue
    grep -q -- "$MULTIARCH_INC" "$pc" && continue
    sed -i "/^Cflags:/ s|\$| -I$MULTIARCH_INC|" "$pc"
  done
  grep -h "^Cflags" "$ROOT"/usr/lib/x86_64-linux-gnu/pkgconfig/*.pc | sort -u | sed 's/^/    /'
fi

echo
echo "═══ 验证 ═══"
export PATH="$ROOT/usr/bin:$PATH"
export M4="$ROOT/usr/bin/m4"
export BISON_PKGDATADIR="$ROOT/usr/share/bison"
export PKG_CONFIG_PATH="$ROOT/usr/lib/x86_64-linux-gnu/pkgconfig:$ROOT/usr/share/pkgconfig"
ok=1
for t in flex bison m4 mkimage dtc; do
  if command -v "$t" >/dev/null; then printf '  ✅ %-9s %s\n' "$t" "$(command -v $t)"; else printf '  ✗ %-9s 缺失\n' "$t"; ok=0; fi
done
printf '  ✅ %-9s %s\n' "flex ver" "$(flex --version 2>&1)"
printf '  ✅ %-9s %s\n' "bison" "$(bison --version 2>&1 | head -1)"
printf '  ✅ %-9s %s\n' "mkimage" "$(mkimage -V 2>&1 | head -1)"
echo "  ── bison 的两个坑（必须靠环境变量，Debian 写死了 /usr）──"
echo "     M4=$M4"
echo "     BISON_PKGDATADIR=$BISON_PKGDATADIR"
echo "  ── libcrypto（certs/extract-cert 需要）──"
if pkg-config --exists libcrypto; then
  echo "     ✅ pkg-config libcrypto: cflags='$(pkg-config --cflags libcrypto)' libs='$(pkg-config --libs libcrypto)'"
else
  echo "     ✗ pkg-config 找不到 libcrypto —— 构建会在 certs/extract-cert 处失败"; ok=0
fi
# 真正编一个用到 openssl 的小程序，确认头文件 + 链接都通
T=$(mktemp -d)
cat > "$T/t.c" <<'EOF'
#include <openssl/bio.h>
#include <openssl/evp.h>
int main(void){ BIO *b = BIO_new(BIO_s_mem()); BIO_free(b); return 0; }
EOF
if gcc "$T/t.c" -o "$T/t" $(pkg-config --cflags --libs libcrypto) 2>"$T/err"; then
  echo "     ✅ openssl 头文件 + 链接实测通过"
else
  echo "     ✗ openssl 编译实测失败："; sed 's/^/       /' "$T/err" | head -5; ok=0
fi
rm -rf "$T"

echo
[ "$ok" = 1 ] && echo "✅ 主机工具链就绪。构建请用 scripts/build/host-tools-env.sh" \
              || { echo "✗ 有缺项，见上"; exit 1; }
