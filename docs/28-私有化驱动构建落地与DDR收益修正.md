# 28 · ★★★ 私有化驱动构建落地：自己编的内核已上板并验收通过

> 承接 docs/27（环境攻坚与构建性能重构）。本篇是**结果篇**：
> 自编内核已部署到板子、成功启动、NPU 与网络全部正常。
>
> 这一步是用户明确设的门槛：「当前还没有做到充分逆向吧，起码也得先实现私有化的驱动构建再说」。
> **门已过。**

```
内核 release   6.1.99-rockchip-rk3588-priv1
上板验收       通过 18 / 警告 1 / 失败 0
NPU 实跑       yolov5s 跑通，中位 37.55 ms（3 核，DDR 未钉频）
NPU 驱动版本   RKNPU driver: v0.9.8   （debugfs 实测，与原厂一致）
```

---

## 1. 为什么这一步值得单独记

「私有化驱动构建」不只是"能编出来"。它的真正门槛是：

**你得能证明自己编的这个内核和原厂那个是等价的 —— 否则后面所有实验的基线都不成立。**

所以这次不是"编出来就完事"，而是把等价性做成了可验证的证据链：

| 证据 | 方法 | 结果 |
|---|---|---|
| **设备树完全相同** | `sha256sum` 与原厂 `/boot/dtb/.../rk3588-orangepi-5-plus.dtb` 对比 | ✅ **逐字节一致**（`86220ae7…`） |
| **NPU 驱动源码完全相同** | `diff -rq` 内核树 `drivers/rknpu/` vs 板子在跑的源码 | ✅ **逐字节一致**（diff 为空） |
| **内核镜像规模相同** | 与原厂 `vmlinuz` 比字节数 | ✅ 均 41,402,880 字节 |
| **配置差异可枚举** | `diff` 相对 `.config.vendor` | ✅ 仅 8 行，全部是有意的（关 DEBUG_INFO + pahole 版本号） |
| **驱动运行时版本一致** | `cat /sys/kernel/debug/rknpu/version` | ✅ `RKNPU driver: v0.9.8` |
| **模块可加载** | 穷尽符号解析体检（见下） | ✅ 272 个模块全部可解析 |

有了这条链，"新内核上 NPU 行为不同"这个变量就被排除掉了 —— 后面任何差异都只能来自我们的插桩。

## 2. 最容易被忽略的致命检查：模块符号解析

这是本次唯一一个**真有可能导致板子失联**的隐患，必须讲清楚。

### 问题的由来

为了省时间我们**只编了 `Image`，没跑 `make modules`**。而顶层 `Module.symvers` 是 `make modules`
阶段生成的 —— 不跑它，modpost 就不知道 vmlinux 导出了哪些符号，于是把**所有**外部符号都报成：

```
WARNING: modpost: "ieee80211_stop_tx_ba_session" [rtw89_core.ko] undefined!
WARNING: modpost: "_printk" [fuse.ko] undefined!
…
```

第一次构建 77 条，第二次 11 条。**这些绝大多数是假警报**：`CONFIG_MODVERSIONS` 没开 →
加载时不校验 CRC → 只要符号真被内核导出，运行时就能解析。

但"绝大多数是假警报"不等于"全是假警报"。如果 rtw89 真有解析不了的符号，装上去加载失败
→ **WiFi 起不来 → 板子失联 → 只能人工断电**。这种事不能赌，必须事前验证。

### 我先用错了方法（记下来）

第一版验证我用 `readelf -p __ksymtab_strings vmlinux` 提取内核导出符号名，得到 13,610 个，
然后判出 **159 个"无法解析"** —— 里面居然有 `_printk`、`call_rcu`、`complete`、`device_add`
这种内核最基础的符号。这显然不对，但结果看起来"很像真的"。

查下去发现：**那个字符串节漏了约 600 个符号名**（`_printk` 就是漏掉的之一，
而源码里明明写着 `kernel/printk/printk.c:2361: EXPORT_SYMBOL(_printk);`）。

权威做法是看 `nm` 里的 `__ksymtab_<name>` 条目：

```bash
nm vmlinux | awk '/^[0-9a-f]+ r __ksymtab_/{sub(/^[0-9a-f]+ r __ksymtab_/,""); print}' | sort -u
# → 14204 个，一个不少
```

> **教训：别用字符串节去反推符号表。** 用 `nm` 的符号条目。

### 正确方法下的迭代收敛

体检逻辑：`未定义符号 − 内核导出 − 同期模块互供 = 真正无法解析的`，并对板子实测的
10 个必需模块做白名单判定。

| 迭代 | MODDIRS | 无法解析 | 关键模块 |
|---|---|---|---|
| 1 | 8 个目录（rtw89 / r8125 / pgdrv / hwmon / fuse / overlayfs / ipv4-netfilter / sched） | 49（全是 `nf_nat_*`/`nft_*`） | ✅ 全干净 |
| 2 | + `net/netfilter`、`net/netfilter/ipset` | 11（剩下全是 IPv6 netfilter） | ✅ 全干净 |
| 3 | + `net/ipv6/netfilter` | **0** | ✅ 全干净 |

**根因**：netfilter 这一支的依赖闭包**不是平的** —— `net/ipv4/netfilter` 里的模块依赖
`net/netfilter` 里的模块（`nf_conntrack`/`nf_nat`/`nf_tables`），再往上还依赖
`net/ipv6/netfilter`。我之前基于"板子加载的 10 个模块闭包是平的"推广到了整个目录，
这个推广是错的。

体检脚本落成 `scripts/build/verify-modules.sh`（退出码：0 全好 / 2 只有附加模块受影响 / 1 关键模块受影响）。
**它现在是部署前的强制门禁。**

## 3. 部署设计：安装与切换分离，回退路径经过演练

### 符号链接双启动

`/boot/boot.scr` 里写死了三个固定名字（`Image`、`dtb/${fdtfile}`、`uInitrd`），
所以用**改符号链接**来切换内核，完全不碰 U-Boot：

```
Image    -> vmlinuz-<release>
dtb      -> dtb-<release>            （目录）
uInitrd  -> uInitrd-<release>
```

`kernel-select` 首次运行会把原厂指向记进 `/boot/.kernel-orig.targets`，
`kernel-select orig` 一键回退。两个内核的文件同时存在，互不影响。

### ⚠ 踩到的坑：`update-initramfs` 会偷改符号链接

`install-kernel.sh` 的设计原则是"只新增带版本号的文件，绝不碰启动符号链接"。
但实际上板后发现：切换到新内核**之前**，`/boot/uInitrd` 就已经指向 `-priv1` 了。

原因：板子上有 Orange Pi 的 initramfs post-update hook，`update-initramfs` 结束时它自动
"Converting to u-boot format" 生成 `uInitrd-<release>`，**并且顺手改写了 `/boot/uInitrd` 符号链接**。
于是系统进入一个"半切换"状态（uInitrd 新、Image/dtb 旧）。

后果很实际：`kernel-select` 记录"原厂指向"时把 `uInitrd` 记成了 `-priv1`，
**这条回退路径是脏的** —— 真要回退时，原厂内核会配上一个只含 272 个模块的新 initramfs。

处置：切换前显式把 `/boot/.kernel-orig.targets` 修正为三条原厂值，并**把回退动作打印出来演练过一遍**：

```
ln -sfn /boot/vmlinuz-6.1.99-rockchip-rk3588    /boot/Image
ln -sfn /boot/dtb-6.1.99-rockchip-rk3588        /boot/dtb
ln -sfn /boot/uInitrd-6.1.99-rockchip-rk3588    /boot/uInitrd
```

> **教训：回退路径必须在"还不需要它"的时候就演练。** 等出事再验证回退，就晚了。

### 挂根不需要任何模块

部署前专门确认过：根文件系统在 `/dev/mmcblk0p1`（eMMC）+ ext4，而这两个驱动在新内核里都是
**内置**（`MMC_DW_ROCKCHIP=y`、`EXT4_FS=y`）。所以即使 initramfs 出问题，挂根这条线也不依赖模块。
initramfs 里那 62 个模块（usb 网卡、ppp、xfs…）是厂商顺手塞的大杂烩，不是挂根所必需。

## 4. ★ 修正 docs/25：DDR 钉频的真实收益是 1.59×，不是 1.8~2.5×

### 错误的方法给出了**反向**的结论

装完钉频后我做了个 A/B，结果是：

```
performance(DDR 2400) 中位: 30.11 ms
dmc_ondemand(默认 534) 中位: 27.89 ms      ← 对照组更快？！
```

也就是"钉频反而更慢"，且与 docs/27 里 verify 那轮的 37.55 ms 也对不上。

**这个结论是错的，错在方法**：

1. `echo ondemand > governor` **不保证频率真的降下来**。上一步刚把 min/max 顶到 2400，
   切回 ondemand 后它未必会掉回 534 —— 所谓"对照组"很可能还在 2400 上跑，**两组其实没差异**，
   剩下的 2.2 ms 差异纯粹是漂移。
2. 顺序测量把温度/缓存/后台负载的漂移混进了组间差异。
3. 样本太少，而这块板子波动很大（docs/25 记载过同一实验得到 25.5~55.0 ms）。

### 严格重测（`scripts/board/bench-ddr-ab.sh`）

规矩沿用 docs/25 定下的：**软钉 min_freq=max_freq 并每步读回确认** + **A/B 交错** + **多轮多样本**。

```
低档 534 MHz  vs  高档 2400 MHz   （8 轮交错，每轮 30 次推理，NPU 固定 1GHz）

第1轮  534: 43.19 ms | 2400: 27.76 ms
第2轮  534: 47.76 ms | 2400: 29.13 ms
第3轮  534: 46.61 ms | 2400: 29.19 ms
第4轮  534: 44.64 ms | 2400: 28.70 ms
第5轮  534: 50.11 ms | 2400: 28.64 ms
第6轮  534: 45.38 ms | 2400: 29.79 ms
第7轮  534: 50.37 ms | 2400: 28.61 ms
第8轮  534: 43.02 ms | 2400: 30.71 ms

中位:  534 MHz = 46.00 ms     2400 MHz = 28.91 ms
提升:  1.59×   （差 17.09 ms）
```

两组**完全不重叠**（43.0~50.4 vs 27.8~30.7），每轮的实际频率都读回确认过。这才是可信的数。

### 结论怎么改

| | 旧结论（docs/25） | 修正后 |
|---|---|---|
| DDR 钉频收益 | 约 2×（1.8~2.5×） | **1.59×** |
| 定性判断 | DRAM 带宽是主要瓶颈 | **仍然成立** |

**定性结论不变、定量修正**：DRAM 带宽确实是最大的单点瓶颈（1.59× 远大于多核调度的 1.28×），
"做算子融合以减少 DRAM 往返"这条技术路线依然是对的方向。只是收益预期要从 2× 调到 1.6×。

> **教训（第二次记同一条）**：测性能差异时，**"对照组"必须被显式确认过**，
> 不能靠 governor 的名字去假设它的状态。这个方法错误这次给出的不是"偏大/偏小"，
> 而是**方向都反了** —— 如果不复核，就会得出"钉频没用"的错误结论并把优化方向带偏。

## 5. 上板验收明细

`scripts/board/verify-kernel.sh`（7 段 19 项，一条命令判生死）：

```
1. 内核版本        ✅ release 匹配
2. NPU 驱动        ✅ /dev/dri/renderD129 在位
                   ✅ debugfs version: RKNPU driver: v0.9.8
                   ✅ 平台驱动已绑定 fdab0000.npu
                   ✅ devfreq: 1000000000 Hz
3. WiFi（命门）    ✅ wlP2p33s0 存在 / rtw89 4 个模块已加载
                   ✅ IP 192.168.31.128/24 / 网关可达
                   ✅ r8125 有线网卡驱动已加载
4. SSH             ✅ 22 端口监听 / ssh.service active
5. 风扇温控        ✅ 风扇驱动在位，6 个 thermal zone 均 36~37°C
6. 系统健康        ✅ 看门狗 1min29s / net-watchdog active+enabled / 无 panic
7. NPU 实跑        ✅ yolov5s 跑通，中位 37.55 ms（3核）
```

唯一警告是"dmesg 里没有 rknpu 记录"—— 内核 cmdline 带 `loglevel=1`，日志本来就安静，非故障。

## 6. 当前板子状态

```
内核           6.1.99-rockchip-rk3588-priv1（自编，已上板运行）
启动方式       /boot/{Image,dtb,uInitrd} 符号链接 → -priv1；
               原厂内核文件完好，kernel-select orig 可一键回退（回退动作已演练）
NPU            performance @ 1GHz
DDR/DMC        performance @ 2400MHz（rknpu-perf.service 开机自启）
看门狗         systemd RuntimeWatchdogSec=60 → 实际生效 1min29s
网络自愈       net-watchdog active（端到端双检：网关 + 开发机）
风扇           正常，36~37°C
```

## 7. 下一步

私有化构建这条线已闭合。接下来才是用户真正要的**逆向 NPU 结构与指令集、最大化利用率**：

1. **叠第一批驱动插桩**（作为相对"已验证等价"基线的显式 diff，出问题能立刻归因）：
   - per-submit 计时 / 计数器（`hw_elapse_time` 已有，补提交侧统计）
   - per-core busy/idle 暴露 —— 解释为什么"2 核只有 1.05×"
   - per-submit DRAM 带宽计数器（action 13/14/15/16/17）—— 把 1.59× 这个带宽瓶颈**量化到字节**
2. **DDR 频率与 NPU 负载联动**：现在是常钉 2400MHz，改成 NPU 忙时拉高、闲时放开
   （省电 + 不影响性能），这是插桩之后的第一个可落地的优化
3. 回到指令集/流水本身：把 docs/18 的段布局、docs/20 的分块结构、docs/24 的调度逻辑
   合成一个能**预测给定工作量在 NPU 上的耗时**的模型 —— 有 1.59× 这条带宽约束打底，
   模型才有物理意义

---

*本篇的复核方式*：等价性靠 sha256/diff/字节数三条硬证据；模块可加载性靠穷尽符号解析
（14204 个导出符号为基准）；DDR 收益靠 8 轮交错实测（每轮读回确认频率）。
`scripts/build/verify-modules.sh` 与 `scripts/board/bench-ddr-ab.sh` 都可直接复跑。
