#!/usr/bin/env python3
"""re_stack.py — 「通用逆向工具流」的统一入口：能力盘点 + 任务路由 + 两个独立实现

## 为什么需要它

本项目的逆向长期靠**手搓脚本**（pyelftools 解析 + capstone 反汇编 + 裸 gdb）。
手搓不是问题，问题是：**同一个人写的"提取器"和"验证器"会共享同一个错误**（docs/142 教训 4）。
所以这里做两件事：

1. **capability** —— 把本机/板端已有的通用工具**实测**一遍，输出能力矩阵。
   不问"业界有什么"，只问"这台机器现在能用什么"。
2. **route** —— 输入一个未解问题，输出该走哪条工具链（而不是一句"用 Ghidra"）。

另外提供两个**与既有脚本不同算法**的实现，用来交叉验证自己的结论：

- `vtables`：两跳指针搜索（name 串 → typeinfo 对象 → vtable 槽）定位 vtable，
  **不依赖任何硬编码的段偏移**（rtti_find.py 的 KNOWN 表是按架构写死的，容易踩）。
- `elf`：用 LIEF（不同于项目其它地方的 pyelftools）重述 ELF 事实。

## 分层模型（业界事实标准，不是本项目的私有约定）

    L0 元信息/探针       file / readelf / strings / checksec
    L1 静态反汇编+反编译  Ghidra / IDA / Binary Ninja / rizin(radare2) / RetDec / Snowman
    L2 类型与符号恢复     RTTI 脚本 / BSim(FLIRT) / DWARF / BTF / pahole
    L3 动态插桩与追踪     gdb(+pwndbg/GEF) / Frida / uprobes / ftrace / rr / eBPF
    L4 仿真与指令级       Unicorn / Qiling / QEMU-TCG-plugin / gem5
    L5 符号执行/约束      angr / Triton / Miasm / z3
    L6 差分与模糊测试     AFL++ / honggfuzz / libFuzzer / syzkaller / 差分对拍
    L7 硬件寄存器级       devmem / /dev/mem / regmap-debugfs / ftrace-regmap / JTAG

## 用法
  python3 src/tools/re_stack.py capability [--board]
  python3 src/tools/re_stack.py route [关键词...]
  python3 src/tools/re_stack.py elf <file>
  python3 src/tools/re_stack.py vtables <file> [--filter RX] [--json OUT] [--limit N]
  python3 src/tools/re_stack.py ghidra status
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys

HOME = os.path.expanduser("~")
TOOLS = os.path.join(HOME, "rknn-work", "tools")
BOARD = "orangepi@192.168.31.128"
BOARD_KEY = os.path.join(HOME, ".ssh", "keys", "opi_board")

# ────────────────────────────────────────────────────────────────────────────
# 环境探测
# ────────────────────────────────────────────────────────────────────────────


def which(name: str, extra_dirs=()) -> str | None:
    p = shutil.which(name)
    if p:
        return p
    for d in extra_dirs:
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def run(cmd, timeout=15, env=None) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def has_py(mod: str) -> tuple[bool, str]:
    try:
        m = __import__(mod)
        return True, str(getattr(m, "__version__", "") or "")
    except Exception:
        return False, ""


def board_sh(script: str, timeout=30) -> str:
    """在板端执行一段 shell。SSH 约三成概率卡在 kex，故重试。"""
    base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", "-o", "IPQoS=none",
            "-o", "ServerAliveInterval=4", "-o", "ServerAliveCountMax=2",
            "-i", BOARD_KEY, BOARD, "bash -s"]
    last = ""
    for _ in range(5):
        try:
            r = subprocess.run(base, input=script, capture_output=True, text=True, timeout=timeout)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            continue
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
        last = (r.stderr or "").strip()[:80]
    return ""


# ────────────────────────────────────────────────────────────────────────────
# 能力矩阵
# ────────────────────────────────────────────────────────────────────────────

# (键, 层, 说明, 本项目里它能干什么)
LOCAL_SPEC = [
    ("ghidra",        "L1", "免费反编译器，AARCH64 完备，headless 可脚本化", "67 个 pass 的 C 伪码；RTTI→类恢复"),
    ("rizin",         "L1", "radare2 的现代分支，轻量、aarch64 原生", "快速 xref / 函数边界"),
    ("radare2",       "L1", "老牌 CLI 框架", "同上（apt 里是 5.5.0，偏老）"),
    ("retdec",        "L1", "学术反编译器，C 输出可读", "对照 Ghidra 的第二意见"),
    ("snowman",       "L1", "轻量反编译器，插件式", "对照"),
    ("aarch64-objdump","L1", "交叉 binutils，无需运行环境即可反汇编", "任何 aarch64 ELF 的即时反汇编"),
    ("gdb-multiarch", "L3", "跨架构 gdb，可 attach / remote", "板端进程与板端 core 的调试"),
    ("frida",         "L3", "运行时插桩，aarch64 支持好", "替代 gdb 做函数级 hook（无内核权限也行）"),
    ("rr",            "L3", "录制-重放，确定性重跑", "一次推理跑一遍，之后随便倒退"),
    ("afl-fuzz",      "L6", "覆盖率引导模糊测试", "librknnrt 的 API 面 / 编译器输入"),
    ("honggfuzz",     "L6", "另一套模糊器，支持硬件反馈", "同上"),
    ("angr",          "L5", "CFG 恢复 + 符号执行", "反推 tiling 判定条件"),
    ("triton",        "L5", "指令级 concolic", "单条路径的约束反解"),
    ("miasm",         "L5", "IR + 符号执行 + 反编译", "轻量替代 angr"),
    ("qiling",        "L4", "基于 Unicorn+QEMU 的全系统仿真", "跑 aarch64 用户态不依赖板子"),
    ("unicorn",       "L4", "CPU 指令仿真（纯库）", "**给 NPU 指令流写解释器**"),
    ("capstone",      "L1", "反汇编库（项目已用）", "所有自制反汇编的底座"),
    ("lief",          "L1", "ELF/PE/MachO 解析与改写", "结构事实的独立来源"),
    ("keystone",      "L4", "汇编器库", "造测试指令流"),
    ("z3",            "L5", "SMT 求解", "配合 angr/Triton"),
    ("pyelftools",    "L1", "ELF 解析（项目已用）", "现有脚本底座"),
]

BOARD_SPEC = [
    ("ftrace(function)",  "L3", "CONFIG_FUNCTION_TRACER=y，可抓调用图与耗时", "**驱动 85 个 rknpu_* 符号的调用树**"),
    ("ftrace(function_graph)", "L3", "带层级与时长", "定位 submit→irq 的真实路径与耗时占比"),
    ("uprobes",           "L3", "CONFIG_UPROBES=y", "用户态 librknnrt 函数级追踪（无内核符号也行）"),
    ("kprobes",           "L3", "**内核未开启**", "❌ 无法对内核做动态探针"),
    ("perf",              "L3", "perf_event_paranoid=2", "用户态采样可行，内核采样需 root"),
    ("bpftrace",          "L3", "未安装；且无 kprobes", "价值有限"),
    ("devmem",            "L7", "未安装（busybox 可能有）", "直接读写 NPU MMIO 寄存器"),
    ("regmap-debugfs",    "L7", "未挂载/无权限", "寄存器读写全量 trace"),
    ("kallsyms",          "L2", "rknpu_* 符号在 kallsyms 里可见", "驱动函数名→地址，**不用反编译驱动**"),
    ("BTF",               "L2", "**无 /sys/kernel/btf/vmlinux**", "❌ 没有免费的内核类型信息"),
    ("gdb",               "L3", "**未安装**", "板端调试需交叉 gdb + gdbserver"),
]


def probe_local() -> list[dict]:
    rows = []
    for key, layer, desc, use in LOCAL_SPEC:
        avail, ver = False, ""
        path = ""
        if key == "ghidra":
            p = os.path.join(TOOLS, "ghidra", "ghidra_11.4.2_PUBLIC", "support", "analyzeHeadless")
            avail = os.path.isfile(p)
            path = p
            ver = "11.4.2" if avail else ""
        elif key in ("aarch64-objdump", "gdb-multiarch"):
            p = os.path.join(TOOLS, "root", "usr", "bin", key if key != "aarch64-objdump"
                             else "aarch64-linux-gnu-objdump")
            avail = os.path.isfile(p)
            path = p if avail else ""
            ver = ""
            if avail:
                env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(TOOLS, "root", "usr", "lib", "x86_64-linux-gnu"))
                _, out = run([p, "--version"], env=env)
                m = re.search(r"(\d+\.\d+(?:\.\d+)?)", out)
                ver = m.group(1) if m else ""
        elif key in ("capstone", "lief", "keystone", "pyelftools"):
            mod = {"pyelftools": "elftools"}.get(key, key)
            avail, ver = has_py(mod)
        elif key == "z3":
            avail, ver = has_py("z3")
        elif key in ("angr", "triton", "miasm", "qiling", "unicorn", "frida", "rr"):
            avail, ver = has_py(key)
            if not avail:
                p = which(key)
                if p:
                    avail, path = True, p
        elif key == "afl-fuzz":
            p = which("afl-fuzz")
            avail, path = bool(p), p or ""
        elif key == "honggfuzz":
            p = which("honggfuzz")
            avail, path = bool(p), p or ""
        else:
            p = which(key)
            avail, path = bool(p), p or ""
        rows.append(dict(key=key, layer=layer, ok=avail, version=ver, path=path, desc=desc, use=use))
    return rows


def probe_board() -> list[dict]:
    script = r'''
echo "KALLSYMS=$(grep -c ' rknpu' /proc/kallsyms 2>/dev/null)"
echo "FTRACE=$(grep -m1 -c '' /sys/kernel/tracing/available_tracers 2>/dev/null)"
echo "FUNCS=$(zcat /proc/config.gz 2>/dev/null | grep -c 'CONFIG_FUNCTION_TRACER=y')"
echo "DYNF=$(zcat /proc/config.gz 2>/dev/null | grep -c 'CONFIG_DYNAMIC_FTRACE=y')"
echo "TRACEFS=$([ -d /sys/kernel/tracing ] && echo 1 || echo 0)"
N=$(grep -c ' rknpu' /proc/kallsyms 2>/dev/null); echo "NKALL=$N"
echo "KPROBE=$(zcat /proc/config.gz 2>/dev/null | grep -c 'CONFIG_KPROBES=y')"
echo "UPROBE=$(zcat /proc/config.gz 2>/dev/null | grep -c 'CONFIG_UPROBES=y')"
echo "BTF=$([ -e /sys/kernel/btf/vmlinux ] && echo 1 || echo 0)"
echo "KPTR=$(cat /proc/sys/kernel/kptr_restrict 2>/dev/null)"
echo "PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)"
for t in gdb gdbserver perf bpftrace devmem busybox python3; do
  command -v $t >/dev/null 2>&1 && echo "HAS_$t=1" || echo "HAS_$t=0"
done
'''
    out = board_sh(script, timeout=40)
    kv = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    def pick(key):
        spec = dict((k, (l, d, u)) for k, l, d, u in BOARD_SPEC)
        layer, desc, use = spec.get(key, ("?", "", ""))
        ok = False
        ver = ""
        if key.startswith("ftrace"):
            ok = kv.get("FUNCS") == "1" and kv.get("TRACEFS") == "1"
            ver = f"dynftrace={kv.get('DYNF','?')}"
        elif key == "uprobes":
            ok = kv.get("UPROBE") == "1"
        elif key == "kprobes":
            ok = False
            ver = "CONFIG_KPROBES not set" if kv.get("KPROBE") == "0" else ""
        elif key == "kallsyms":
            ok = int(kv.get("KALLSYMS", "0") or 0) > 0
            ver = f"{kv.get('NKALL','?')} 个 rknpu_* 符号"
        elif key == "BTF":
            ok = kv.get("BTF") == "1"
            ver = "无 BTF" if not ok else "有 BTF"
        elif key == "perf":
            ok = kv.get("HAS_perf") == "1"
            ver = f"paranoid={kv.get('PARANOID','?')}"
        elif key == "bpftrace":
            ok = kv.get("HAS_bpftrace") == "1"
        elif key == "devmem":
            ok = kv.get("HAS_devmem") == "1" or kv.get("HAS_busybox") == "1"
        elif key == "regmap-debugfs":
            ok = False
            ver = "debugfs 需 root"
        elif key == "gdb":
            ok = kv.get("HAS_gdb") == "1" or kv.get("HAS_gdbserver") == "1"
        return dict(key=key, layer=layer, ok=ok, version=ver, desc=desc, use=use)

    if not kv:
        return [dict(key=k, layer=l, ok=None, version="板端不可达", desc=d, use=u)
                for k, l, d, u in BOARD_SPEC]
    return [pick(k) for k, _, _, _ in BOARD_SPEC]


def fmt_matrix(rows, title):
    print(f"\n{title}")
    print("  " + "-" * 100)
    for r in rows:
        if r["ok"] is None:
            mark = "  ?"
        else:
            mark = " ✓" if r["ok"] else " ✗"
        ver = f" ({r['version']})" if r.get("version") else ""
        print(f"  {mark} [{r['layer']}] {r['key']:<22}{ver}")
        print(f"        {r['desc']}")
        print(f"        → {r['use']}")
    ok = sum(1 for r in rows if r["ok"])
    print("  " + "-" * 100)
    print(f"  可用 {ok}/{len(rows)}")


# ────────────────────────────────────────────────────────────────────────────
# 任务路由
# ────────────────────────────────────────────────────────────────────────────

ROUTES = [
    {
        "kw": ["类", "class", "vtable", "虚表", "RTTI", "pass", "分裂", "继承"],
        "q": "从 strip 过的 C++ .so 里恢复类结构、虚函数、调用关系",
        "chain": [
            ("L1", "Ghidra headless `-import` + `RecoverClassesFromRTTI.java`",
             "官方实现，AARCH64 完备；输出类名/父类/虚函数表/交叉引用"),
            ("L2", "`src/compiler/rtti_find.py`（自研，硬编码段偏移）**仅作对照**",
             "两者结果不一致时，以 Ghidra 为准；自研那份用于快速单类查询"),
            ("L1", "`aarch64-linux-gnu-objdump -d` 只用来确认 vtable 里的地址确实是函数序言",
             "最便宜的一票否决"),
        ],
        "why": "本项目的 67 个 pass 全部是模板类，手写解析器已踩过一次段偏移错误",
    },
    {
        "kw": ["成本", "cost", "性能", "计时", "耗时", "预算", "预测", "模型"],
        "q": "预测某个子图在 NPU 上的耗时",
        "chain": [
            ("L6", "**差分对拍**：同构链 N=1..8 拟合边际成本 + 固定开销",
             "这是唯一在本项目里真正奏效的办法（docs/140）"),
            ("L0", "编译器 IR 表（verbose=True）的 Cycles 列",
             "**不能直接用**：系统性乐观 4.7×，但可以做相对排序"),
            ("L4", "Unicorn 复现 NPU 指令流后计时（若指令集逆到可执行）",
             "远期：唯一能给出 cycle 级的路径"),
        ],
        "why": "解析式三次都栽在跨尺寸失效；先做分档查表 + 插值，别再做解析式",
    },
    {
        "kw": ["驱动", "driver", "内核", "kernel", "ioctl", "submit", "寄存器", "段", "边界"],
        "q": "搞清内核驱动/寄存器的行为（提交路径、IRQ、寄存器语义）",
        "chain": [
            ("L2", "`/proc/kallsyms` 里 85 个 `rknpu_*` 符号",
             "**驱动不是黑盒**：函数名全部可见，无需反编译即可定位"),
            ("L3", "ftrace function_graph 抓 `rknpu_*`",
             "CONFIG_FUNCTION_TRACER=y、无 kprobes 也够用；拿调用树 + 每段耗时"),
            ("L3", "uprobes（CONFIG_UPROBES=y）跟用户态 `ioctl(fd=…)`",
             "把「用户态哪个调用」和「驱动哪个函数」对上"),
            ("L7", "`devmem` 读 NPU MMIO（需 root，且要确认没有 IOMMU 拦截）",
             "寄存器语义的最终裁决"),
        ],
        "why": "板子没开 kprobes、没 BTF，但**有符号 + 有 ftrace**，这条路完全没被用过",
    },
    {
        "kw": ["指令", "instruction", "ISA", "指令集", "流水", "pipeline", "task", "regcmd"],
        "q": "还原 NPU 的指令编码与流水",
        "chain": [
            ("L4", "Unicorn 写 aarch64/RISC-V 假想 ISA 的解释器",
             "先假设编码，再用手上的 regcmd dump 回归验证"),
            ("L6", "差分：改一个字段 → 看 IR 表/计时怎么变",
             "唯一能确认字段语义的手段（单变量原则）"),
            ("L1", "Ghidra 反编译 `librknnrt` 里的 task 打包函数",
             "拿到写入顺序与位域"),
        ],
        "why": "`DPU(4097).16416`、`f08/f0c/f18` 这类未解字段只能靠单变量差分",
    },
    {
        "kw": ["对拍", "一致性", "精度", "参考", "cosine", "diff", "验证"],
        "q": "证明某个子模型/整链的数值正确性",
        "chain": [
            ("L6", "**权威参考实现**（上游 PyTorch/diffusers）+ 逐层 cosine",
             "纪律：参考绝不能是自研的改写版（CLIP 那次同源同错）"),
            ("L6", "逐层中间张量落盘（npz）+ 逐层比对",
             "定位第一处发散，而不是只看最终输出"),
        ],
        "why": "docs/142 教训 4",
    },
    {
        "kw": ["存根", "stub", "hook", "拦截", "追踪", "trace", "运行"],
        "q": "在不改二进制的前提下观察运行时行为",
        "chain": [
            ("L3", "LD_PRELOAD 存根（最轻，已验证有效）",
             "只拦得到动态符号；内部静态函数无效"),
            ("L3", "Frida（aarch64 支持好，能 hook 任意地址）",
             "比 gdb 轻，可脚本化，适合反复跑"),
            ("L3", "gdb + gdbserver（重，但能看栈和寄存器）",
             "只在需要精确现场时用"),
        ],
        "why": "本项目已证明：布局观测靠 hook 拿不到，但符号级拦截能拿到",
    },
]


def cmd_route(args):
    q = " ".join(args.words) if args.words else ""
    if not q:
        print("\n可用路由：")
        for r in ROUTES:
            print(f"  {'/'.join(r['kw'][:4]):<28} {r['q']}")
        print("\n用法：python3 src/tools/re_stack.py route 成本    （或用任意中文/英文关键词）")
        return
    hits = [r for r in ROUTES if any(k.lower() in q.lower() for k in r["kw"])]
    if not hits:
        print(f"未命中「{q}」。可用关键词示例：类/成本/驱动/指令/对拍/hook")
        return
    for r in hits:
        print(f"\n{'='*78}\n问题：{r['q']}\n{'='*78}")
        for i, (layer, tool, why) in enumerate(r["chain"], 1):
            print(f"  {i}. [{layer}] {tool}")
            print(f"       {why}")
        print(f"\n  判定依据：{r['why']}")


# ────────────────────────────────────────────────────────────────────────────
# ELF 事实（LIEF 实现，独立于项目其它地方的 pyelftools）
# ────────────────────────────────────────────────────────────────────────────

import math


def shannon(b: bytes) -> float:
    if not b:
        return 0.0
    freq = [0] * 256
    for c in b:
        freq[c] += 1
    n = len(b)
    return -sum((f / n) * math.log2(f / n) for f in freq if f)


def cmd_elf(args):
    import lief
    b = lief.parse(args.path)
    if b is None:
        sys.exit(f"LIEF 无法解析 {args.path}")
    print(f"\n文件: {args.path}")
    print(f"  格式      {b.header.file_type}  {b.header.machine_type}")
    print(f"  入口      {b.entrypoint:#x}")
    print(f"  PIE       {b.is_pie}    NX {b.has_nx}   RELRO {b.get(lief.ELF.DynamicEntry.TAG.BIND_NOW) is not None}")
    print(f"\n  段/节（SHF_ALLOC）:")
    print(f"    {'name':<22}{'vaddr':>12}{'size':>10}{'熵':>7}  用途")
    total = 0
    for s in b.sections:
        try:
            if not (int(s.flags) & 0x2):   # SHF_ALLOC
                continue
        except Exception:
            pass
        data = bytes(s.content)
        if not data:
            continue
        total += len(data)
        try:
            kind = "代码" if (int(s.flags) & 0x4) else "数据"
        except Exception:
            kind = "?"
        print(f"    {s.name:<22}{s.virtual_address:>#12x}{s.size:>10}{shannon(data[:1 << 20]):>7.2f}  {kind}")
    print(f"\n  映射总量  {total/1e6:.2f} MB")

    syms = list(b.symbols) + list(b.dynamic_symbols)
    funcs = [s for s in syms if s.type == lief.ELF.Symbol.TYPE.FUNC and s.value]
    print(f"  符号      {len(syms)} 个（FUNC {len(funcs)}）")
    demangled = [s for s in syms if s.name.startswith("_Z")]
    print(f"  C++ 修饰名 {len(demangled)} 个")
    try:
        libs = sorted({l.name for l in b.libraries})
    except Exception:
        libs = sorted({e.name for e in b.dynamic_entries
                       if type(e).__name__ == "DynamicEntryLibrary"})
    print(f"  依赖库    {', '.join(libs) or '（无）'}")

    # 调试信息线索
    for s in b.sections:
        if "debug" in s.name or "gnu_debuglink" in s.name:
            print(f"  ⚠ {s.name}: {bytes(s.content)[:80]!r}")


# ────────────────────────────────────────────────────────────────────────────
# RTTI / vtable（两跳指针搜索；不依赖硬编码段偏移）
# ────────────────────────────────────────────────────────────────────────────

def _demangle_simple(m: str) -> str:
    """够用的 Itanium 嵌套名还原：N4onnc10CustomPassI...E → onnc::CustomPass<...>"""
    s = m
    if s.startswith("_Z"):
        s = s[2:]
    if not s.startswith("N"):
        s = re.sub(r"^\d+", "", s)
        return s
    i, out = 1, []
    while i < len(s):
        if s[i] == "E":
            break
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            n = int(s[i:j])
            if n == 0:
                i = j
                continue
            out.append(s[j:j + n])
            i = j + n
        elif s[i] == "I":
            out.append("<")
            inner = s[i + 1:]
            parts, k = [], 0
            while k < len(inner):
                if inner[k].isdigit():
                    jj = k
                    while jj < len(inner) and inner[jj].isdigit():
                        jj += 1
                    ln = int(inner[k:jj])
                    parts.append(inner[jj:jj + ln])
                    k = jj + ln
                elif inner[k] == "E":
                    break
                else:
                    k += 1
            out.append(", ".join(parts))
            out.append(">")
            break
        else:
            i += 1
    return "::".join(x for x in out if x)


def cmd_vtables(args):
    import lief
    b = lief.parse(args.path)
    if b is None:
        sys.exit(f"LIEF 无法解析 {args.path}")

    segs = []
    for s in b.sections:
        try:
            if not (int(s.flags) & 0x2):
                continue
        except Exception:
            pass
        data = bytes(s.content)
        if data:
            segs.append((s.virtual_address, s.virtual_address + len(data), data, s.name))
    segs.sort()

    exec_ranges = [(lo, hi) for lo, hi, _, n in segs if n.startswith((".text", ".plt", ".init", ".fini"))]

    def read_va(va, n):
        for lo, hi, data, _ in segs:
            if lo <= va < hi:
                return data[va - lo: va - lo + n]
        return b""

    def is_code(va):
        return any(lo <= va < hi for lo, hi in exec_ranges)

    # 只对「真的会存指针」的段建索引。
    # ★ 必须排除 .rela.*：重定位表的 addend 字段长得和指针一模一样，
    #   不排除会把每条重定位都当成一个 vtable 槽（踩过：45 个槽变 0 个类）。
    PTR_SECS = (".data.rel.ro", ".data", ".got", ".got.plt", ".init_array", ".fini_array")
    idx = {}
    for lo, hi, data, name in segs:
        if name not in PTR_SECS:
            continue
        n = (len(data) // 8) * 8
        for off in range(0, n, 8):
            v = int.from_bytes(data[off:off + 8], "little")
            if 0x1000 < v < 0x1_0000_0000:
                idx.setdefault(v, []).append(lo + off)

    # 候选类名字符串
    cand = []
    pat = re.compile(rb"(?:N[0-9][0-9A-Za-z_]{6,300}E|[0-9]{1,3}[A-Z][0-9A-Za-z_]{4,300})\x00")
    for lo, hi, data, name in segs:
        if name not in (".rodata", ".data.rel.ro", ".data"):
            continue
        for m in pat.finditer(data):
            cand.append((lo + m.start(), m.group()[:-1].decode("latin1")))
    if args.filter:
        cand = [c for c in cand if re.search(args.filter, c[1], re.I)]

    # 两跳：name 串 → typeinfo(+8 处持有 name) → vtable 槽(持有 typeinfo)
    pairs = []          # (vtable_slot, typeinfo, name_va, mangled)
    for sva, mangled in cand:
        for q in idx.get(sva, []):
            if q == sva - 8:          # name 字段自身
                continue
            ti = q - 8
            for vs in idx.get(ti, []):
                pairs.append((vs, ti, sva, mangled))
                break

    # ── vtable 边界：不能只靠"已知类名"来定，否则未知类的 vtable 会被并进前一个。
    # 直接按**结构**扫 .data.rel.ro：某个槽的值 v 若满足
    #   v+8 也指向同区、且 v+8 指向的字节像一个 mangled 名 → v 就是一个 typeinfo 对象，
    #   该槽就是 vtable 的 typeinfo 槽。
    def looks_like_name(va):
        raw = read_va(va, 300)
        if len(raw) < 4:
            return False
        nul = raw.find(b"\x00")
        if nul < 4:
            return False
        txt = raw[:nul]
        return bool(re.fullmatch(rb"[A-Za-z_][0-9A-Za-z_.$]{3,250}", txt)) and \
            bool(re.match(rb"(_Z)?(N[0-9]|[0-9])", txt) or b"_" in txt)

    slot_set = set()
    for lo, hi, data, name in segs:
        if name not in PTR_SECS:
            continue
        n = (len(data) // 8) * 8
        for off in range(0, n, 8):
            a = lo + off
            v = int.from_bytes(data[off:off + 8], "little")
            if not (0x1000 < v < 0x1_0000_0000):
                continue
            if looks_like_name(v + 8):
                slot_set.add(a)
    slot_lo = sorted(slot_set)

    rows = {}
    for vs, ti, sva, mangled in pairs:
        ap = vs + 8
        nxt = next((x for x in slot_lo if x > vs), None)
        # 布局：[offset-to-top][typeinfo] = vs，紧接着 n 个函数指针，
        # 下一个 vtable 的 typeinfo 槽在 vs + 16 + 8n ⇒ n = (nxt - vs - 16)//8
        limit = (nxt - 8) if nxt is not None else ap + 256 * 8
        fns = []
        a = ap
        while a + 8 <= limit and len(fns) < 256:
            v = int.from_bytes(read_va(a, 8), "little")
            if not is_code(v):
                break
            fns.append(v)
            a += 8
        if not fns:
            continue
        truncated = len(fns) >= 64
        cur = rows.get(mangled)
        if cur is None or len(fns) > cur["n_virtual"]:
            rows[mangled] = dict(mangled=mangled, demangled="", name_va=sva, typeinfo=ti,
                                 vtable_slot=vs, addr_point=ap, n_virtual=len(fns),
                                 truncated=truncated, first_fn=fns[0], fns=fns)

    rows = sorted(rows.values(), key=lambda r: (-r["n_virtual"], r["name_va"]))
    for r in rows:
        r["demangled"] = _demangle(r["mangled"])

    print(f"\n{args.path}")
    print(f"  候选 mangled 名 {len(cand)} 个 → 解析出 vtable 的类 {len(rows)} 个"
          f"（vtable 槽总数 {len(pairs)}）")
    print(f"\n  {'虚函数':>5}  {'vtable@':>10}  {'typeinfo@':>10}  类名")
    print("  " + "-" * 100)
    for r in rows[: args.limit]:
        n = f"{r['n_virtual']}" + ("+" if r.get("truncated") else "")
        print(f"  {n:>5}  {r['addr_point']:>#10x}  {r['typeinfo']:>#10x}  {r['demangled']}")
    if len(rows) > args.limit:
        print(f"  ... 其余 {len(rows)-args.limit} 个省略（--limit 调整）")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\n  已写出 {args.json}")

    # 只在参考文件与目标同一二进制时对照
    stem = os.path.basename(args.path)
    refdir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "analysis"))
    for cand_ref in ("librknnc_rtti.txt", "librknnrt_vtables.txt"):
        ref = os.path.join(refdir, cand_ref)
        if not os.path.isfile(ref):
            continue
        if "c" in os.path.basename(ref) and "librknnc" in stem:
            pass
        elif "librknnc" not in stem:
            continue
        with open(ref) as f:
            old = {l.strip() for l in f if l.strip()}
        new = {r["mangled"] for r in rows}
        print(f"\n  与 {cand_ref} 对照：旧 {len(old)} / 新 {len(new)} / 交集 {len(old & new)}"
              f" / 仅旧有 {len(old-new)} / 仅新有 {len(new-old)}")


_CPPFILT = None


def _demangle(m: str) -> str:
    """优先用 binutils 的 c++filt（权威实现），退化到内置的简易还原。"""
    global _CPPFILT
    if _CPPFILT is None:
        for c in (shutil.which("c++filt"),
                  os.path.join(TOOLS, "root", "usr", "bin", "x86_64-linux-gnu-c++filt"),
                  os.path.join(TOOLS, "root", "usr", "bin", "aarch64-linux-gnu-c++filt")):
            if c and os.path.isfile(c):
                _CPPFILT = c
                break
        _CPPFILT = _CPPFILT or ""
    if _CPPFILT:
        rc, out = run([_CPPFILT, m], timeout=10)
        if rc == 0 and out.strip() and not out.startswith("_Z"):
            return out.strip()
    return _demangle_simple(m)


# ────────────────────────────────────────────────────────────────────────────
# Ghidra 包装
# ────────────────────────────────────────────────────────────────────────────

def cmd_ghidra(args):
    gp = os.path.join(TOOLS, "ghidra", "ghidra_11.4.2_PUBLIC")
    hl = os.path.join(gp, "support", "analyzeHeadless")
    jdk = os.path.join(TOOLS, "jdk21")
    print(f"\nGhidra Home : {gp}  {'✓' if os.path.isdir(gp) else '✗'}")
    print(f"analyzeHeadless: {hl}  {'✓' if os.path.isfile(hl) else '✗'}")
    print(f"JDK         : {jdk}  {'✓' if os.path.isdir(jdk) else '✗'}")
    if os.path.isfile(os.path.join(jdk, "bin", "javac")):
        print(f"javac       : ✓（Ghidra 需要 JDK 而非 JRE）")
    for name in ("RecoverClassesFromRTTIScript.java",):
        p = os.path.join(gp, "Ghidra", "Features", "Decompiler", "ghidra_scripts", name)
        print(f"{name}: {'✓' if os.path.isfile(p) else '✗'}")
    if args.sub == "run":
        pass


# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="通用逆向工具流：能力盘点 / 任务路由 / 独立实现")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("capability", help="实测本机与板端可用工具")
    p.add_argument("--board", action="store_true", help="同时探测板端（走 SSH）")

    p = sub.add_parser("route", help="按问题路由到工具链")
    p.add_argument("words", nargs="*")

    p = sub.add_parser("elf", help="ELF 事实（LIEF 实现）")
    p.add_argument("path")

    p = sub.add_parser("vtables", help="RTTI→vtable→虚函数（两跳指针搜索）")
    p.add_argument("path")
    p.add_argument("--filter", default="", help="按类名正则过滤")
    p.add_argument("--json", default="", help="结果落盘")
    p.add_argument("--limit", type=int, default=40)

    p = sub.add_parser("ghidra", help="Ghidra 安装状态")
    p.add_argument("sub", nargs="?", default="status")

    args = ap.parse_args()
    if args.cmd == "capability":
        fmt_matrix(probe_local(), "【本机】通用逆向工具（实测）")
        if args.board:
            fmt_matrix(probe_board(), "【板端 Orange Pi 5 Plus】可观测能力（实测）")
    elif args.cmd == "route":
        cmd_route(args)
    elif args.cmd == "elf":
        cmd_elf(args)
    elif args.cmd == "vtables":
        cmd_vtables(args)
    elif args.cmd == "ghidra":
        cmd_ghidra(args)


if __name__ == "__main__":
    main()
