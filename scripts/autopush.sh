#!/usr/bin/env bash
# autopush.sh — 定时提交并推送到 GitHub，防止成果只留在本地
#
# 设计取舍：
#   · 只在**有改动**时提交，不产生空提交噪音
#   · 提交信息带时间戳，便于回溯
#   · 推送失败（断网/认证失效）不删本地提交，下次会连着一块推上去
#   · 全程写日志到 /tmp/m_rk_npu_driver.log，便于事后核查
#
# 安装（用户级 cron，不需要 sudo）：
#   */20 * * * * /bin/bash "/home/dunaandone/orange pi/rk3588-npu-re/scripts/autopush.sh" >>/tmp/m_rk_npu_driver.log 2>&1
set -uo pipefail

REPO="/home/dunaandone/orange pi/rk3588-npu-re"
LOG_TAG="[$(date '+%F %T')]"

# ── 单实例锁：上一次推送若仍在进行，直接跳过本次（防推包叠加打满机器）──
LOCK="/tmp/m_rk_npu_driver.autopush.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$LOG_TAG 上一次推送仍在进行，跳过本次"
  exit 0
fi
BRANCH="main"

cd "$REPO" || { echo "$LOG_TAG ❌ 仓库目录不存在: $REPO"; exit 1; }

# ── 有改动才干活 ──
if [ -z "$(git status --porcelain)" ]; then
  echo "$LOG_TAG 无改动，跳过"
  # 但仍检查是否有未推送的提交（上次推送失败过）
  if git rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
    ahead=$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)
    if [ "$ahead" = "0" ]; then exit 0; fi
    echo "$LOG_TAG 有 $ahead 个未推送提交，准备补推"
  else
    echo "$LOG_TAG 远端分支未建立，准备首次推送"
  fi
else
  n=$(git status --porcelain | wc -l)
  echo "$LOG_TAG 检测到 $n 处改动，提交中…"
  git add -A
  # 生成能看出改了什么的提交信息
  summary=$(git diff --cached --stat | tail -1)
  # ⚠️ 这个 cron 会和「我手写的描述性提交」抢同一个工作树：
  #    如果它在我 `git add -A` 之后、`git commit` 之前触发，我的提交信息就没了。
  #    （踩过一次：docs/128 的成果被记成 "auto: 2026-10-13 10:20:01"）
  #    所以至少把**最新的文档名**写进信息里，保住「这一批落了什么」这条线索。
  newest_doc=$(git diff --cached --name-only --diff-filter=A -- 'docs/*.md' \
               | sort -V | tail -1)
  [ -n "$newest_doc" ] && summary="$summary | 新文档: $(basename "$newest_doc")"
  git commit -q -m "auto: $(date '+%F %T')

$summary" || { echo "$LOG_TAG ⚠️ 提交失败"; exit 1; }
  echo "$LOG_TAG ✅ 已提交: $summary"
fi

# ── 推送 ──
if timeout 600 nice -n 10 ionice -c2 -n7 git -c pack.threads=2 -c pack.windowMemory=128m push -q origin "$BRANCH" 2>&1; then
  echo "$LOG_TAG ✅ 已推送 → $(git rev-parse --short HEAD)"
else
  echo "$LOG_TAG ⚠️ 推送失败（网络或认证问题），提交已留在本地，下次会重试"
  exit 1
fi
