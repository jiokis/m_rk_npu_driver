set pagination off
set confirm off
set width 0
set print elements 0
catch load librknnc.so
run
python
import gdb, re

class DumpBP(gdb.Breakpoint):
    """命中时打印 this 的 N 个 quadword，并把每个像指针的槽位解引用出 4 个 quadword"""
    def __init__(self, spec, off, qw, bt, stop, cap):
        super().__init__(spec, internal=False)
        self.off, self.qw, self.bt, self.stop, self.cap = off, qw, bt, stop, cap
        self.n = 0
        self.silent = True

    def stop(self):
        self.n += 1
        if self.n > self.cap:
            self.enabled = False
            return False
        inf = gdb.selected_inferior()
        rdi = int(gdb.parse_and_eval("(unsigned long)$rdi"))
        out = [f"\n=== HIT off={self.off:#x} this={rdi:#x} (hit #{self.n}) ==="]
        try:
            mem = bytes(inf.read_memory(rdi, 8 * self.qw))
        except Exception as e:
            out.append(f"  [read this failed: {e}]")
            mem = b""
        vals = []
        for i in range(len(mem) // 8):
            v = int.from_bytes(mem[8 * i:8 * i + 8], "little")
            vals.append(v)
            out.append(f"  [{i:2d}] {v:#018x}")
        for i, v in enumerate(vals):
            if v < 0x10000 or v > 0x7FFFFFFFFFFF:
                continue
            try:
                m2 = bytes(inf.read_memory(v, 32))
            except Exception:
                out.append(f"    [{i}] -> {v:#x}: <不可读>")
                continue
            w = [int.from_bytes(m2[8 * k:8 * k + 8], "little") for k in range(4)]
            out.append(f"    [{i}] -> {v:#x}: " + " ".join(f"{x:#x}" for x in w))
        gdb.write("\n".join(out) + "\n")
        if self.bt:
            gdb.write(gdb.execute(f"bt {self.bt}", to_string=True))
        gdb.flush()
        return self.stop

out = gdb.execute("info sharedlibrary librknnc", to_string=True)
m = re.search(r"(0x[0-9a-f]+)\s+0x", out)
if not m:
    gdb.write("[probe] 解析 sharedlibrary 失败\n")
else:
    text_load = int(m.group(1), 16)
    base = text_load - 0x11b260
    gdb.write(f"[probe] text_load={text_load:#x}  base={base:#x}\n")
    DumpBP('*' + hex(base + 0x2cece0), 0x2cece0, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x2cfd90), 0x2cfd90, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x2d0e40), 0x2d0e40, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x2e8240), 0x2e8240, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x2e8250), 0x2e8250, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x2e8260), 0x2e8260, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x3913e0), 0x3913e0, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x391400), 0x391400, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x391410), 0x391410, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x391480), 0x391480, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x391510), 0x391510, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x39b010), 0x39b010, 6, 5, False, 40)
    DumpBP('*' + hex(base + 0x39d4f0), 0x39d4f0, 6, 5, False, 40)
end
continue
printf "\n[probe] run finished\n"
quit
