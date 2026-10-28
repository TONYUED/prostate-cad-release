#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把本仓库发布到 GitHub（**不需要本机安装 Git**）。

原理：直接调用 GitHub REST API 的 Git 数据库接口
    blobs -> tree -> commit -> refs/heads/main
在一次提交内写入全部仓库文件，等价于 `git init && git add -A && git commit`。

前置条件
--------
1. GitHub 个人访问令牌（PAT），需要 `repo` 作用域：
   https://github.com/settings/tokens  ->  Generate new token (classic)
   勾选 repo（若只是公开仓库仍需 repo 才能创建仓库）。
2. 通过环境变量传入，避免令牌写入文件/历史：
       PowerShell: $env:GITHUB_TOKEN = "ghp_xxx"
       bash:       export GITHUB_TOKEN=ghp_xxx

用法
----
    # 先看看会上传哪些文件（不联网、不需要令牌）
    python scripts/publish_to_github.py --dry-run

    # 创建私有仓库并推送
    python scripts/publish_to_github.py --repo prostate-cad-gleason --private

    # 创建公开仓库、打标签并发布 Release
    python scripts/publish_to_github.py --repo prostate-cad-gleason --public --tag v1.0.0

    # 推送到已存在的仓库（仓库已存在时自动复用）
    python scripts/publish_to_github.py --owner <你的用户名> --repo <仓库名> --private

注意
----
- 数据与产物目录（data/ 除 gold_standards.json 外、outputs/、results/、legacy/）
  不会被上传，规则与 .gitignore 保持一致。
- 若仓库已存在且非空，脚本会拒绝覆盖，避免误删远程历史。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime

API = "https://api.github.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 .gitignore 保持一致
SKIP_DIRS = {
    ".git", "__pycache__", "data", "outputs", "results",
    ".smoke", "legacy", ".venv", "venv", "env",
}
SKIP_EXTS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svs",
    ".ndpi", ".bmp", ".pt", ".pth", ".onnx", ".engine",
    ".pyc", ".pyo", ".zip",
}


def api(method: str, path: str, token: str, data: dict | None = None):
    url = API + path
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "prostate-cad-publisher")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(
            "[GitHub API 错误] %s %s -> %d\n%s" % (method, path, e.code, detail[:800])
        )


def collect_files(root: str) -> list:
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
        rel_dir = os.path.relpath(dp, root).replace("\\", "/")
        for f in sorted(fn):
            if os.path.splitext(f)[1].lower() in SKIP_EXTS:
                continue
            rel = f if rel_dir == "." else rel_dir + "/" + f
            out.append(rel)
    return sorted(out)


def main() -> int:
    p = argparse.ArgumentParser(
        description="发布本仓库到 GitHub（无需本机 Git）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--repo", required=True, help="仓库名")
    p.add_argument("--owner", default=None, help="仓库所有者，默认取令牌对应用户")
    p.add_argument("--token", default=None, help="PAT；缺省读环境变量 GITHUB_TOKEN")
    p.add_argument("--branch", default="main")
    p.add_argument("--private", action="store_true", help="创建私有仓库")
    p.add_argument("--public", action="store_true", help="创建公开仓库（默认）")
    p.add_argument("--tag", default=None, help="同时打标签并创建 Release，如 v1.0.0")
    p.add_argument("--commit-message", default=None)
    p.add_argument("--dry-run", action="store_true", help="只列出将上传的文件")
    args = p.parse_args()

    files = collect_files(ROOT)
    if not files:
        raise SystemExit("[错误] 没有找到可上传的文件。")

    print("=" * 66)
    print("发布到 GitHub")
    print("=" * 66)
    print(f"仓库  : {args.owner or '<自动>'}/{args.repo}")
    print(f"分支  : {args.branch}")
    print(f"可见性: {'private' if args.private else 'public'}")
    print(f"文件数: {len(files)}")
    print(f"根目录: {ROOT}\n")

    for f in files:
        print(f"  {f}")

    if args.dry_run:
        print("\n[dry-run] 未联网，未做任何改动。")
        return 0

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit(
            "[错误] 未提供 PAT。请设置环境变量 GITHUB_TOKEN，或用 --token 传入。\n"
            "       生成地址: https://github.com/settings/tokens （勾选 repo）"
        )

    me = api("GET", "/user", token)
    owner = args.owner or me.get("login")
    if not owner:
        raise SystemExit("[错误] 无法解析 GitHub 用户名，请用 --owner 指定。")

    # 仓库已存在则复用；不存在则创建
    try:
        repo = api("GET", "/repos/%s/%s" % (owner, args.repo), token)
        print(f"\n[信息] 使用已存在仓库: {repo['html_url']}")
        if repo.get("size", 0) > 0:
            raise SystemExit(
                "[中止] 远程仓库非空，为避免覆盖历史，请手动处理或换一个仓库名。"
            )
    except SystemExit as e:
        msg = str(e)
        if "404" in msg:
            repo = api(
                "POST", "/user/repos", token,
                {
                    "name": args.repo,
                    "description": "Reproducible pipeline for Gleason pattern segmentation on SICAPv2 (code only).",
                    "private": bool(args.private),
                    "auto_init": False,
                    "has_issues": True,
                    "has_wiki": False,
                },
            )
            print(f"\n[信息] 已创建仓库: {repo['html_url']}")
        else:
            raise

    # 1) blobs
    tree = []
    for rel in files:
        with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
            content = f.read()
        blob = api("POST", "/repos/%s/%s/git/blobs" % (owner, args.repo), token,
                   {"content": content, "encoding": "utf-8"})
        tree.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        print(f"  blob {rel}")

    # 2) tree
    created_tree = api("POST", "/repos/%s/%s/git/trees" % (owner, args.repo), token,
                       {"tree": tree})
    print(f"\n[信息] tree {created_tree['sha']}")

    # 3) commit
    message = args.commit_message or (
        "Initial release v1.0.0: Gleason pattern segmentation pipeline (code only)\n\n"
        "- gold standards derived from SICAPv2 official masks (244 cores)\n"
        "- U-Net semantic segmentation training / sliding-window inference\n"
        "- three-level evaluation: pixel Dice/IoU, core-level grade agreement, ROC\n"
        "- image data not included (SICAPv2 license)"
    )
    commit = api("POST", "/repos/%s/%s/git/commits" % (owner, args.repo), token,
                 {"message": message, "tree": created_tree["sha"], "parents": []})
    print(f"[信息] commit {commit['sha']}")

    # 4) ref
    api("POST", "/repos/%s/%s/git/refs" % (owner, args.repo), token,
        {"ref": "refs/heads/" + args.branch, "sha": commit["sha"]})
    print(f"[信息] branch {args.branch} -> {commit['sha']}")

    if args.tag:
        api("POST", "/repos/%s/%s/git/refs" % (owner, args.repo), token,
            {"ref": "refs/tags/" + args.tag, "sha": commit["sha"]})
        api("POST", "/repos/%s/%s/releases" % (owner, args.repo), token,
            {
                "tag_name": args.tag,
                "name": args.tag,
                "body": (
                    "Code release for the Gleason pattern segmentation pipeline on SICAPv2.\n\n"
                    "Includes: gold-standard derivation, dataset preparation, U-Net training,\n"
                    "sliding-window inference, and three-level evaluation.\n"
                    "Image data are not included; obtain SICAPv2 from its official source."
                ),
                "draft": False,
                "prerelease": False,
            })
        print(f"[信息] release {args.tag} 已创建")

    print("\n[完成]")
    print(f"  {repo['html_url']}")
    print(f"  {repo['clone_url']}")
    print(f"  提交时间: {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
