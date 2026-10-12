#!/bin/bash
# 模块符号解析体检：确认每个 .ko 的未定义符号都能被【内核导出】或【同期模块】满足
#
# ═══ 为什么需要这个检查（这是本次构建最容易被忽略的坑）═══
#
# 我们为了省时间**只编了 Image，没跑 `make modules`**。而顶层 `Module.symvers` 是
# `make modules` 阶段生成的 —— 不跑它，modpost 就不知道 vmlinux 导出了哪些符号，
# 于是把**所有**外部符号都报成 `WARNING: modpost: "xxx" [...] undefined!`。
#
# 这些警告绝大多数是**假警报**：`CONFIG_MODVERSIONS` 没开 → 加载时不校验 CRC →
# 只要符号真的被内核导出，运行时就能解析。但如果真有符号既不被内核导出、
# 也不由同期模块提供，那这个 .ko **装上去也加载不了** —— 而如果它是 rtw89，
# 板子直接失联，得人工断电才能救。所以必须事前验证，不能赌。
#
# ★ 判定导出符号的正确姿势（踩过坑）：
#   一开始我用 `readelf -p __ksymtab_strings vmlinux` 提取导出符号名，
#   结果**漏了约 600 个**（连 `_printk`、`call_rcu` 这种最基础的都漏了），
#   于是误判出 159 个"无法解析"。真正权威的做法是看 `nm` 里的 `__ksymtab_<name>`
#   条目 —— vmlinux 里有 14204 个，一个不少。
#   教训：**别用字符串节去反推符号表，用 nm 的符号条目。**
#
# 用法：scripts/build/verify-modules.sh [modstage目录]
set -uo pipefail

ROOT=/home/dunaandone/rk-kernel
S="$ROOT/kernel"
OUT="${1:-$ROOT/out}"
MODSTAGE="$OUT/modstage"
NM="$(command -v aarch64-none-linux-gnu-nm || command -v nm)"

# 板子实际加载的模块（上板 lsmod 实测得到，是"绝对不能出问题"的白名单）
CRITICAL="rtw89_core rtw89_8852b rtw89_8852be rtw89_pci pwm-fan r8125 pgdrv fuse ip_tables sch_fq_codel overlay cuse"

echo "═══ 模块符号解析体检 ═══"
echo "  模块目录: $MODSTAGE"
[ -d "$MODSTAGE" ] || { echo "✗ 找不到 $MODSTAGE" >&2; exit 1; }
[ -e "$S/vmlinux" ] || { echo "✗ 找不到 $S/vmlinux（先构建 Image）" >&2; exit 1; }

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
find "$MODSTAGE" -name '*.ko' | sort > "$T/kos"
N=$(wc -l < "$T/kos")
echo "  模块数:   $N"
echo "  工具:     $NM"

# ── 1) 内核导出的符号（权威：nm 的 __ksymtab_<name> 条目）──
"$NM" "$S/vmlinux" 2>/dev/null \
  | awk '/^[0-9a-f]+ r __ksymtab_/{sub(/^[0-9a-f]+ r __ksymtab_/,""); print}' \
  | sort -u > "$T/exported"
echo "  内核导出: $(wc -l < "$T/exported") 个符号"

# ── 2) 未定义符号 / 模块之间互相提供的符号 ──
: > "$T/undef"; : > "$T/provided"
while read -r k; do
  "$NM" -u "$k" 2>/dev/null | awk '{print $NF}'
done < "$T/kos" | sort -u > "$T/undef"
while read -r k; do
  "$NM" --defined-only "$k" 2>/dev/null | awk '{print $NF}'
done < "$T/kos" | sort -u > "$T/provided"
echo "  未定义:   $(wc -l < "$T/undef") 个（去重）"
echo "  模块互供: $(wc -l < "$T/provided") 个"

# ── 3) 真正无法解析的 ──
comm -23 "$T/undef" <(sort -u "$T/exported" "$T/provided") > "$T/unresolved"

if [ ! -s "$T/unresolved" ]; then
  echo
  echo "  ✅✅ 全部 $N 个模块的未定义符号都能解析 —— 可以安全部署"
  exit 0
fi

echo
echo "  ⚠ 有 $(wc -l < "$T/unresolved") 个符号无法解析："
sed 's/^/      /' "$T/unresolved" | head -30
[ "$(wc -l < "$T/unresolved")" -gt 30 ] && echo "      …（其余省略）"

# ── 4) 哪些模块受影响，有没有踩到关键白名单 ──
echo
echo "  受影响的模块："
: > "$T/bad"
while read -r k; do
  if "$NM" -u "$k" 2>/dev/null | awk '{print $NF}' | grep -qxF -f "$T/unresolved"; then
    echo "      ${k##*/}"
    echo "$k" >> "$T/bad"
  fi
done < "$T/kos"

echo
BAD_CRIT=0
for m in $CRITICAL; do
  f=$(find "$MODSTAGE" -name "$m.ko" | head -1)
  [ -n "$f" ] || continue
  if grep -qxF "$f" "$T/bad"; then
    echo "  ✗✗ 关键模块 $m 受影响 —— **不要部署**"
    BAD_CRIT=1
  fi
done

echo
if [ "$BAD_CRIT" = 1 ]; then
  cat <<'EOF'
  ✗✗ 关键模块（板子启动/联网必需）不可加载，绝对不能上板。
     处置：把这些符号的来源目录也加进 build-kernel.sh 的 MODDIRS 一起编。
     常见缺的目录：net/netfilter（nf_conntrack/nf_nat/nf_tables）、net/netfilter/ipset
EOF
  exit 1
fi

echo "  ⚠ 关键模块都干净，受影响的只是用不到的附加模块。"
echo "     建议还是把缺的目录补进 MODDIRS 一起编（半个模块集是未来的坑）："
echo "       net/netfilter  net/netfilter/ipset"
exit 2
