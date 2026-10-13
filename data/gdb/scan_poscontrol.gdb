set pagination off
set confirm off
set width 0
set breakpoint pending on
catch load librknnc.so
run
break RKNNCompiler_build
continue
finish
python
import gdb, re, struct


def rw_regions(pid):
    out = []
    for ln in open(f"/proc/{pid}/maps"):
        p = ln.split()
        if len(p) < 2 or "w" not in p[1]:
            continue
        a, b = p[0].split("-")
        lo, hi = int(a, 16), int(b, 16)
        if hi - lo > 0x40000000:      # 跳过离谱的大映射
            continue
        out.append((lo, hi))
    return out


def scan(inf, value, regions):
    pat = struct.pack("<Q", value)
    hits = []
    CH = 0x100000
    for lo, hi in regions:
        off = lo
        while off < hi:
            n = min(CH, hi - off)
            try:
                buf = bytes(inf.read_memory(off, n))
            except Exception:
                off += n
                continue
            i = buf.find(pat)
            while i >= 0:
                hits.append(off + i)
                i = buf.find(pat, i + 1)
            off += n
    return hits


def qw(inf, addr, n):
    try:
        b = bytes(inf.read_memory(addr, 8 * n))
    except Exception:
        return None
    return [int.from_bytes(b[8 * k:8 * k + 8], "little") for k in range(n)]


def main():
    vt_off = 0x1e05d60
    dump_off, dump_len, deref = 0, 0x10, 4
    out = gdb.execute("info sharedlibrary librknnc", to_string=True)
    m = re.search(r"(0x[0-9a-f]+)\s+0x", out)
    if not m:
        gdb.write("[scan] 解析 sharedlibrary 失败\n")
        return
    base = int(m.group(1), 16) - 0x11b260
    target = base + vt_off
    gdb.write(f"[scan] base={base:#x}  vtable={target:#x}\n")
    inf = gdb.selected_inferior()
    pid = inf.pid
    regs = rw_regions(pid)
    gdb.write(f"[scan] 可写内存区 {len(regs)} 个，共 "
              f"{sum(h - l for l, h in regs) / 2**20:.1f} MiB\n")
    hits = scan(inf, target, regs)
    gdb.write(f"[scan] 命中 vtable 指针 {len(hits)} 处\n")
    for h in hits[:20]:
        gdb.write(f"\n=== 控制块 @ {h:#x}  对象 @ {h + 0x10:#x} ===\n")
        cb = qw(inf, h, 4)
        if cb:
            gdb.write("  cb: " + " ".join(f"{x:#x}" for x in cb) + "\n")
        obj = qw(inf, h + 0x10, (dump_off + dump_len) // 8 + 1)
        if not obj:
            gdb.write("  <对象不可读>\n")
            continue
        for i in range(0, dump_off // 8):
            gdb.write(f"  obj[{i:3d}] {obj[i]:#018x}\n")
        for i in range(dump_off // 8, min(len(obj), (dump_off + dump_len) // 8)):
            gdb.write(f"  obj[{i:3d}] {obj[i]:#018x}\n")
        # 把 dump 区间里像指针的槽位解引用
        for i in range(dump_off // 8, min(len(obj), (dump_off + dump_len) // 8)):
            v = obj[i]
            if v < 0x10000 or v > 0x7FFFFFFFFFFF:
                continue
            w = qw(inf, v, deref)
            if w:
                gdb.write(f"    obj[{i}] -> {v:#x}: " + " ".join(f"{x:#x}" for x in w) + "\n")


main()
end
quit
