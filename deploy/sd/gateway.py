#!/usr/bin/env python3
"""gateway.py — **提示词注入 → 出图 → NPU 利用率** 的网关（跑在板子上）

## 设计约束（板子上只有 PIL + rknnlite，没有 torch/flask/onnxruntime）

* HTTP 用 **标准库 `http.server`**（多线程 ThreadingHTTPServer）
* 分词用 **纯标准库** 的 `clip_tokenizer.py`
* 全部推理走 **RKNN**（CLIP 文本塔 / UNet 三段 / VAE 解码器）
* 不用 numpy 以外的第三方库（numpy 是 rknnlite 的依赖，板上已有）

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 一个极简 HTML 表单（**提示词注入**入口） |
| GET/POST | `/generate?prompt=...&steps=1` | 生成一张图，返回 PNG |
| GET | `/npu` | NPU 利用率（JSON） |
| GET | `/health` | 各模型是否就位 |

## NPU 利用率怎么读

`/sys/class/devfreq/fdab0000.npu/load` → 形如 `100@1000000000Hz`（**百分比 @ 频率**）。
另外 `/sys/class/devfreq/fdab0000.npu/cur_freq` 是当前频率。
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("SD_MODELS", "/tmp")
TOKDIR = os.environ.get("SD_TOKENIZER", "/tmp/clip_tok")
NPU_LOAD = "/sys/class/devfreq/fdab0000.npu/load"
NPU_FREQ = "/sys/class/devfreq/fdab0000.npu/cur_freq"

# ---------------------------------------------------------------- NPU 监控

def npu_load():
    """返回 {"percent": int, "freq_hz": int}；读不到就返回 None"""
    try:
        s = open(NPU_LOAD).read().strip()          # 形如 "100@1000000000Hz"
        pct, _, freq = s.partition("@")
        return {"percent": int(pct), "freq_hz": int(freq.rstrip("Hz"))}
    except Exception:
        return None


class NpuSampler(threading.Thread):
    """后台按固定间隔采 NPU 利用率（出图时才能看到峰值）"""

    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            v = npu_load()
            if v:
                self.samples.append((time.time(), v["percent"]))
                if len(self.samples) > 4000:
                    self.samples = self.samples[-2000:]
            time.sleep(self.interval)

    def window(self, seconds):
        t0 = time.time() - seconds
        v = [p for (t, p) in self.samples if t >= t0]
        return {
            "n": len(v),
            "max": max(v) if v else None,
            "mean": round(sum(v) / len(v), 1) if v else None,
            "now": v[-1] if v else None,
        }


SAMPLER = NpuSampler()

#: SD 的 `scaled_linear` beta 表（1000 步）——SD-Turbo/SD1.5/SD2.1 都用它
def _make_alphas(n=1000, b0=0.00085, b1=0.012):
    betas = np.linspace(b0 ** 0.5, b1 ** 0.5, n) ** 2
    return np.cumprod(1.0 - betas).astype(np.float64)


ALPHAS = _make_alphas()

#: SD 的 VAE `scaling_factor`（**只在解码时除**；解码那步已烘进 ONNX，所以网关里不用碰）
VAE_SCALING = 0.18215


# ★★★★★ SD-Turbo 的**正确调度**：`EulerAncestralDiscreteScheduler`
#   + `timestep_spacing="trailing"` + `prediction_type="epsilon"`，**不是 DDIM**。
#
#   三个我踩过的数：
#   ① **`init_noise_sigma = 14.61`**（不是 1.0！）—— 初始噪声要乘它
#   ② **latent 进 UNet 前不乘 VAE 的 0.18215**（那是解码时才除的；我的 VAE ONNX 已烘进去）
#   ③ DDIM 在 t=999 起步时 `1/sqrt(ᾱ)≈15`，会把 eps 误差放大 15 倍；
#      EulerAncestral 的 1 步更新其实很干净：`z ← z − σ·eps`
def _sigmas(n=1000, b0=0.00085, b1=0.012):
    betas = np.linspace(b0 ** 0.5, b1 ** 0.5, n) ** 2
    ac = np.cumprod(1.0 - betas)
    return np.sqrt((1.0 - ac) / ac)            # epsilon 预测下 σ = sqrt((1-ᾱ)/ᾱ)


SIGMAS = _sigmas()
SIGMA_MAX = float(SIGMAS[-1])                  # ≈14.61


def euler_ancestral_schedule(steps: int):
    """复刻 `set_timesteps(steps)`（trailing 间距）：返回 (timesteps, sigmas)"""
    # ★★★★★ 必须用**截断**（`.astype(int)`）而不是 `round()`！
    #   diffusers 的 `timestep_spacing="trailing"` 实测给 4 步 `[999,749,499,249]`，
    #   而 `round()` 会给 `[999,749,500,250]` ⇒ 第 2/3 步的 timestep 差 1
    #   ⇒ UNet 拿到的 t 不同 ⇒ eps 不同 ⇒ **多步累积出可观的偏差**（踩过）
    ts = np.linspace(0, len(SIGMAS) - 1, steps + 1).astype(int)[::-1][:-1]
    sg = np.concatenate([SIGMAS[ts], [0.0]])
    return ts, sg


def scale_model_input(z, i, sg):
    """★★★★★ **必须做这一步**（diffusers 的 `scheduler.scale_model_input`）

    EulerAncestral 是 `sample / sqrt(σ²+1)`。我原来把 std≈14.6 的 latent **直接**喂进 UNet
    ⇒ UNet 看到的输入大了 14.65 倍 ⇒ 预测的 eps 全是垃圾 ⇒ **出图是抽象彩色纹理**。
    这就是找了很久的真凶；加上之后 latent 的 std 从 16.8 变成 **1.37**（正确量级），
    多步之后出图变成**真实的连贯画面**。
    """
    sigma = float(sg[i])
    return (z / math.sqrt(sigma ** 2 + 1.0)).astype(np.float32)


def euler_step(eps, i, ts, sg, z, rng=None):
    """一步 **EulerDiscrete**（SD-Turbo 的 `model_index.json` 写的就是它，不是 EulerAncestral）

    ★★★★★ 区别：**EulerDiscrete 不注入噪声**（`sigma_up = 0`）
    ⇒ 给定 seed 是**完全确定**的 ⇒ NPU 与 GPU 参考应当**逐像素对上**。
    我之前用 EulerAncestral，每步都加一份随机噪声 ⇒ 两边内容必然不同
    （对比时看到的「背景一致、主体不同」就是它造成的）。

    `x0 = z − σ·eps`；`deriv = (z − x0)/σ`；`z ← z + deriv·(σ_next − σ)`
    """
    sigma, sigma_next = float(sg[i]), float(sg[i + 1])
    x0 = z - sigma * eps
    deriv = (z - x0) / sigma
    return (z + deriv * (sigma_next - sigma)).astype(np.float32)


# ---------------------------------------------------------------- 推理

class Pipeline:
    """三段 UNet + VAE + CLIP，全部 RKNN；用锁串行化（NPU 一次只能跑一个模型）"""

    def __init__(self, models_dir: str):
        self.lock = threading.Lock()
        self._acc_lock = threading.Lock()
        self.acc = {"load": 0.0, "npu": 0.0, "release": 0.0, "calls": 0}
        self.m = models_dir
        self.ready = {}
        # ★★★★★ **常驻模型池**（实测的关键优化）
        #   RKNPU 的 IOVA 空间约 **3.6 GB**（实测：up1 515MB 可驻留 7 个、up3 70MB 可驻留 12 个
        #   ⇒ 是**体积**限制不是个数；11 个模型一起放不下）。
        #   但 **CLIP(3) 与 VAE(2) 是「每张图用一次」，只有 6 个 UNet 段是「每步都用」**
        #   ⇒ 让 **6 个 UNet 段常驻**、CLIP/VAE 按需加载，
        #     每步的「加载→跑→释放」就彻底消失了（原来每步 6 段 × N 步 = 最多 24 次）。
        #   实测单次 load+init+release 的 NPU 侧固定开销 ~0.6 s（与磁盘无关，页缓存后 load 只 0.24 s）
        #   ⇒ 这一步省掉的就是那 25% 的 `model_load_ms`。
        self.resident = {}
        self._preload()
        self.tok = None
        try:
            sys.path.insert(0, HERE)
            from clip_tokenizer import CLIPTokenizer
            self.tok = CLIPTokenizer(os.path.join(TOKDIR, "vocab.json"),
                                     os.path.join(TOKDIR, "merges.txt"))
            self.ready["tokenizer"] = True
        except Exception as e:
            self.ready["tokenizer"] = f"{e}"

    #: 常驻的 6 个 UNet 段
    RESIDENT_MODELS = ("unet_p1.rknn", "unet_p2.rknn", "unet_up0.rknn",
                       "unet_up1.rknn", "unet_up2.rknn", "unet_up3.rknn")

    def _preload(self):
        ok = []
        for n in self.RESIDENT_MODELS:
            try:
                self.resident[n] = self._rknn(n)
                ok.append(n)
            except Exception as e:
                print(f"[gateway] 常驻 {n} 失败: {e}", flush=True)
        print(f"[gateway] 常驻模型 {len(ok)}/{len(self.RESIDENT_MODELS)}: {ok}", flush=True)

    def _rknn(self, name):
        from rknnlite.api import RKNNLite
        r = RKNNLite(verbose=False)
        p = os.path.join(self.m, name)
        if r.load_rknn(p) != 0:
            raise RuntimeError(f"load {p} 失败")
        if r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError(f"init {p} 失败")
        return r

    @staticmethod
    def _nhwc(a):
        return a.transpose(0, 2, 3, 1).astype(np.float32) if a.ndim == 4 and a.shape[2] > 1 else a.astype(np.float32)

    def generate(self, prompt: str, steps: int = 1, seed: int | None = None, size: int = 256):
        from PIL import Image
        t_start = time.time()
        with self._acc_lock:
            self.acc = {"load": 0.0, "npu": 0.0, "release": 0.0, "calls": 0}
        with self.lock:
            timings = {}
            # ① 分词 + CLIP 文本塔
            if self.tok is None:
                raise RuntimeError("tokenizer 未就绪")
            ids = np.array(self.tok(prompt))
            t0 = time.time()
            # ★★★★★ CLIP 的两个关键修正（实测判据）：
            #   ① **词嵌入查表在 numpy 做**（模型里带 `x.long()` 的 int64 Cast，RKNN 处理不了）
            #   ② 三段的导出**不能转置**（`Transformer.batch_first=True`）且**必须带因果掩码**
            #      —— 我原来既转置又漏掩码 ⇒ CLIP 输出与真正值只有 0.266
            te = np.load(os.path.join(self.m, "clip_tok_emb.npy")).astype(np.float32)
            pe = np.load(os.path.join(self.m, "clip_pos_emb.npy")).astype(np.float32)
            x = (te[ids] + pe)[None]
            for nm in ("clipn_p0.rknn", "clipn_p1.rknn", "clipn_p2.rknn"):
                rc = self._rknn(nm)
                x = rc.inference(inputs=[self._nhwc(x)])[0]
                rc.release()
            ehs = x
            timings["clip_ms"] = (time.time() - t0) * 1000

            # ② 潜空间去噪（**真正的 SD 调度**）
            # ★★★★★ 之前写成 `z = z - eps/N` —— 那**不是** DDIM，几乎不改变 z，
            #   所以出图全是初始噪声（实测三张图几乎一样、都是米色噪声）。
            #   SD 用 scaled_linear beta 表；SD-Turbo 只要 1~4 步。
            # ★★★★★ 两个必须做对的缩放（这里错过一次，出图全是噪声）：
            #   ① 初始噪声按调度器的 `init_noise_sigma`（SD 的 EulerAncestral/trailing = **1.0**）
            #   ② **进 UNet 之前要乘 VAE 的 `scaling_factor`（SD = 0.18215）**
            #      diffusers 是 `latents = latents * vae.config.scaling_factor` 才进 UNet；
            #      我原来直接喂 `randn*0.75` ⇒ **UNet 看到的 latent 大了 ~5.5 倍** ⇒ 输出必是噪声。
            #   ③ 解码时 VAE 的 ONNX 里已烘进 `1/scaling_factor`（见 vae_to_rknn.py）⇒ 不用再乘
            L = size // 8
            rng = np.random.RandomState(seed if seed is not None else int(time.time()) % 2**31)
            # ★ 初始噪声按 **init_noise_sigma**（EulerAncestral 实测 = 14.61），不是 1.0
            z = (rng.randn(1, 4, L, L) * SIGMA_MAX).astype(np.float32)
            tlist, siglist = euler_ancestral_schedule(max(steps, 1))
            print(f"[dbg] steps={steps!r} size={size} len(tlist)={len(tlist)} len(siglist)={len(siglist)}", flush=True)
            # ★★★★★ 512 档**不能同时驻留多段**：CMA 只有 1 GB，而 p3a 单个就要 869 MB
            #   ⇒ 同时 load p1+p2+p3a 会 `init 失败`（`RKNPU: failed to allocate IOVA: -12`）。
            #   ⇒ **逐段加载 → 跑 → 立即释放**，任何时刻只驻留一个模型（踩过）。
            # ★★★★★ NPU 利用率实测：把「模型加载」与「NPU 推理」分开计时。
            #   板上没有可信的 NPU load 计数器：
            #     * `/sys/class/devfreq/fdab0000.npu/load` 恒为 `100@1000000000Hz`
            #       （换 `rknpu_ondemand` governor 也一样）
            #     * `/sys/kernel/debug/rknpu/load` 有逐核读数但恒为 0（这版驱动没实现）
            #   ⇒ 从宿主侧量：`npu_busy` = 所有 inference 调用耗时之和，
            #     `util = npu_busy / 墙钟`。**加载/释放模型的时间不算 NPU 占用**，
            #     而这恰恰是当前耗时的大头（逐段加载，见下）。
            def run(name, ins):
                _t0 = time.time()
                # ★ 常驻模型直接用，不加载/释放（这是本轮的关键优化）
                if name in self.resident:
                    _t1 = _t0
                    out = self.resident[name].inference(inputs=[self._nhwc(a) for a in ins])
                    _t2 = time.time()
                    with self._acc_lock:
                        self.acc["npu"] += _t2 - _t1
                        self.acc["calls"] += 1
                    if out is None:
                        raise RuntimeError(f"{name}（常驻）推理返回 None")
                    return out
                r = self._rknn(name)
                _t1 = time.time()
                try:
                    out = r.inference(inputs=[self._nhwc(a) for a in ins])
                except Exception as e:
                    r.release()
                    raise RuntimeError(f"{name} 推理异常: {type(e).__name__}: {e}") from e
                _t2 = time.time()
                r.release()
                _t3 = time.time()
                with self._acc_lock:
                    self.acc["load"] += _t1 - _t0
                    self.acc["npu"] += _t2 - _t1
                    self.acc["release"] += _t3 - _t2
                    self.acc["calls"] += 1
                if out is None:
                    raise RuntimeError(
                        f"{name} 推理返回 None（输入 {len(ins)} 个："
                        f"{[tuple(a.shape) for a in ins]}）—— 多半是输入个数/形状与该 rknn 不符，"
                        f"或 NPU 内存不足（看 dmesg 的 IOVA 报错）")
                return out
            t0 = time.time()
            for i, t in enumerate(tlist):
                t = int(t)
                tb = np.array([float(t)], np.float32)
                zi = scale_model_input(z, i, siglist)          # ★ 关键：喂 UNet 前缩放
                o1 = run("unet_p1.rknn", [zi, tb, ehs])
                # ★ P1 的 `feat` 与 `skip1_6` 是同一个张量 ⇒ RKNN 去重，只返回 8 个
                feat, emb = o1[7], o1[0]
                sk1 = o1[1:8]
                o2 = run("unet_p2.rknn", [feat, emb, ehs])
                mid, sk2 = o2[0], o2[1:]
                # ★★★★★ 512 档把 up_blocks **逐个拆开**（依据逆向出的内存统计：
                #   `Total Weight Memory Size: 825668KB` ⇒ 那笔失败的 869 MB 是**权重**，
                #   不是激活 ⇒ 按参数量拆即可。每段 332/540/164/73 MB，2 GB CMA 装得下）
                sk = list(sk1) + list(sk2)   # 12 条：64,64,64,32,32,32,16,16,16,8,8,8
                h = mid
                # ★★★★★ **变量名必须用 `j`，不能用 `i`！**
                #   外层是 `for i, t in enumerate(tlist)`（采样步），
                #   内层若也写 `for i in range(4)` 会**把采样步序号覆盖成 3**
                #   ⇒ 采样器拿错 σ（`euler_ancestral_step(..., i=3, ...)`），
                #   单步时直接 `IndexError: index 3 is out of bounds for axis 0 with size 2`；
                #   多步时不崩但**每步都用 σ[3]** ⇒ 出图是纹理。这就是找了很久的真凶（踩过）。
                for j in range(4):
                    lo = (4 - 1 - j) * 3
                    # ⚠️ `up_blocks[0]` **没有 cross-attention**（SD 配置是
                    #   `up_block_types=["UpBlock2D","CrossAttnUpBlock2D",...]`）
                    #   ⇒ 导出器把没用到的 `ehs` 剪掉了，up0 只有 5 个输入
                    if j == 0:
                        h = run(f"unet_up{j}.rknn", [h, emb] + sk[lo:lo + 3])[0]
                    else:
                        h = run(f"unet_up{j}.rknn", [h, emb, ehs] + sk[lo:lo + 3])[0]
                eps = h
                z = euler_step(eps, i, tlist, siglist, z)
            timings["unet_ms"] = (time.time() - t0) * 1000

            # ③ VAE 解码
            t0 = time.time()
            # 512 档的 VAE 解码器也是 2 段（整只 198 MB ONNX 转换会 OOM）
            if size == 256:
                rd = self._rknn("sdvae256_fp16.rknn")
                img = rd.inference(inputs=[self._nhwc(z)])[0]
                rd.release()
            else:
                hh = run("vae512_v1.rknn", [z])[0]
                img = run("vae512_v2.rknn", [hh])[0]
            timings["vae_ms"] = (time.time() - t0) * 1000

        # 后处理 → PNG
        a = np.clip((img[0].transpose(1, 2, 0) + 1) / 2 * 255, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="PNG")
        timings["total_ms"] = (time.time() - t_start) * 1000
        # ★ NPU 利用率 = NPU 推理耗时 / 墙钟；同时报出加载/释放占比
        wall = time.time() - t_start
        timings["npu_busy_ms"] = round(self.acc["npu"] * 1000, 1)
        timings["model_load_ms"] = round(self.acc["load"] * 1000, 1)
        timings["npu_calls"] = self.acc["calls"]
        timings["npu_util_pct"] = round(self.acc["npu"] / wall * 100, 1)
        timings["load_pct"] = round(self.acc["load"] / wall * 100, 1)
        return buf.getvalue(), timings


PIPE: Pipeline | None = None

# ---------------------------------------------------------------- HTTP

HTML = """<!doctype html><meta charset=utf-8><title>RK3588 NPU 生图网关</title>
<style>body{font-family:system-ui;max-width:720px;margin:40px auto;padding:0 16px}
input[type=text]{width:100%;padding:10px;font-size:16px}button{padding:10px 18px;font-size:16px}
pre{background:#f4f4f4;padding:12px;border-radius:8px;overflow:auto}img{max-width:100%;margin-top:16px;border-radius:8px}</style>
<h2>RK3588 NPU 生图网关</h2>
<form onsubmit="go(event)">
<input type="text" id=p value="a photo of a cat" placeholder="输入提示词…">
<p>步数 <input type="number" id=s value="1" min="1" max="8" style="width:60px">
   &nbsp; 尺寸 <select id=z><option value=256>256</option><option value=512>512</option></select></p>
<button>生成</button></form>
<div id=out></div>
<script>
async function go(e){e.preventDefault();const o=document.getElementById('out');
o.innerHTML='<pre>生成中…</pre>';
const t0=performance.now();
const r=await fetch('/generate?prompt='+encodeURIComponent(p.value)+'&steps='+s.value+'&size='+z.value);
if(!r.ok){o.innerHTML='<pre>失败: '+await r.text()+'</pre>';return}
const ms=(performance.now()-t0).toFixed(0);
o.innerHTML='<pre>端到端 '+ms+' ms  |  NPU: '+r.headers.get('X-NPU-Load')+'</pre>';
const b=await r.blob();o.innerHTML+='<img src="'+URL.createObjectURL(b)+'">';}
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/plain; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            return self._send(200, HTML, "text/html; charset=utf-8")
        if u.path == "/health":
            return self._send(200, json.dumps({"ready": PIPE.ready if PIPE else None,
                                               "models_dir": MODELS}, indent=1),
                              "application/json")
        if u.path == "/npu":
            return self._send(200, json.dumps({"now": npu_load(),
                                               "last_30s": SAMPLER.window(30)}, indent=1),
                              "application/json")
        if u.path == "/generate":
            prompt = (q.get("prompt") or [""])[0]
            steps = int((q.get("steps") or ["1"])[0])
            size = int((q.get("size") or ["256"])[0])
            try:
                png, tm = PIPE.generate(prompt, steps=steps, size=size)
                return self._send(200, png, "image/png",
                                  {"X-NPU-Load": json.dumps(SAMPLER.window(60)),
                                   "X-Timings": json.dumps(tm)})
            except Exception:
                return self._send(500, "生成失败:\n" + traceback.format_exc())
        self._send(404, "not found")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n).decode() if n else ""
        try:
            d = json.loads(body or "{}")
        except Exception:
            d = dict(urllib.parse.parse_qsl(body))
        if self.path.rstrip("/") == "/generate":
            try:
                png, tm = PIPE.generate(d.get("prompt", ""), steps=int(d.get("steps", 1)),
                                        size=int(d.get("size", 256)),
                                        seed=d.get("seed"))
                return self._send(200, png, "image/png",
                                  {"X-NPU-Load": json.dumps(SAMPLER.window(60)),
                                   "X-Timings": json.dumps(tm)})
            except Exception:
                return self._send(500, "生成失败:\n" + traceback.format_exc())
        self._send(404, "not found")


def main() -> int:
    global PIPE
    port = int(os.environ.get("SD_PORT", "8088"))
    PIPE = Pipeline(MODELS)
    SAMPLER.start()
    print(f"[gateway] 模型目录 {MODELS}   分词器 {TOKDIR}")
    print(f"[gateway] 监听 http://0.0.0.0:{port}/    NPU load 文件 {NPU_LOAD}")
    print(f"[gateway] 初始 NPU: {npu_load()}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
