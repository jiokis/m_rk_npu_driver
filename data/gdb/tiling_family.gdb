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
        # ⚠️ 不能叫 self.stop —— 那会把 Breakpoint.stop() **方法**覆盖成 bool（踩过一次，
        #    报错是 "attribute of type 'bool' is not callable"）
        self.off, self.qw, self.bt, self.do_stop, self.cap = off, qw, bt, stop, cap
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
        return self.do_stop

out = gdb.execute("info sharedlibrary librknnc", to_string=True)
m = re.search(r"(0x[0-9a-f]+)\s+0x", out)
if not m:
    gdb.write("[probe] 解析 sharedlibrary 失败\n")
else:
    text_load = int(m.group(1), 16)
    base = text_load - 0x11b260
    gdb.write(f"[probe] text_load={text_load:#x}  base={base:#x}\n")
    DumpBP('*' + hex(base + 0x284720), 0x284720, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x284730), 0x284730, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x284740), 0x284740, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x284750), 0x284750, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x2bed20), 0x2bed20, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x37ca50), 0x37ca50, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x37eb90), 0x37eb90, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x37eba0), 0x37eba0, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x37ed20), 0x37ed20, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x37ed50), 0x37ed50, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x37ede0), 0x37ede0, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x37fd70), 0x37fd70, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x380b00), 0x380b00, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x380b10), 0x380b10, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x380c90), 0x380c90, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x380cc0), 0x380cc0, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x380d50), 0x380d50, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x3854b0), 0x3854b0, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x3913f0), 0x3913f0, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x391420), 0x391420, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x391450), 0x391450, 4, 4, False, 6)
    DumpBP('*' + hex(base + 0x3914d0), 0x3914d0, 4, 4, False, 6)
end
continue
printf "\n[probe] run finished\n"
quit
