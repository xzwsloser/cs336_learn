---
name: feishu-to-blog
description: 将飞书/Lark Wiki 或 Docx 链接转换为当前 CS336 静态博客的一篇笔记。用户给出 feishu.cn/wiki/...、feishu.cn/docx/... 链接并要求转成博客、笔记、文章，或要求把飞书文档导入本仓库时使用。自动调用 lark-cli 读取正文、下载图片到 assets/images、生成 notes/<slug>.md、更新 notes.json，并校验 Markdown、LaTeX、代码块和图片资源。
---

# Feishu to Blog

把一条飞书文档链接转换成本仓库可发布的一篇博客。这个 skill 只面向当前仓库的静态博客约定：Markdown 源文件放在 `notes/`，图片放在 `assets/images/`，文章索引是 `notes.json`，页面由 `note.html` 使用 marked + KaTeX + highlight.js 渲染。

## 输入与输出

输入是一条飞书文档链接，例如：

```text
https://dcnlcoue52zc.feishu.cn/wiki/SesqwwkKliHovjky4Ehc0WW2noh?from=from_copylink
```

一次成功转换应产生：

- `notes/<slug>.md`：可被 `note.html` 直接渲染的 Markdown。
- `assets/images/<slug>-img-NN.<ext>`：正文引用的本地图片。
- `notes.json` 中恰好多一条对应 slug 的记录。
- 一个通过渲染前检查的结果：无残留飞书 XML 标签、无远程飞书图片 URL、图片文件存在、代码围栏闭合。

## 前置检查

在当前仓库根目录执行。先确认目标确实是本仓库：

```bash
pwd
test -f notes.json && test -f note.html && test -d notes && test -d assets/images
command -v lark-cli
lark-cli whoami
```

如果 `lark-cli` 未登录、token 需要刷新，或沙箱禁止联网/写入 `~/.local/share/lark-cli`，先让用户完成认证或在有权限的环境中重跑。不要用匿名 HTTP 抓取替代 `lark-cli`，否则图片和权限内容会缺失。

## 一键转换

优先使用确定性脚本，不要手工复制正文：

```bash
python3 skills/feishu-to-blog/scripts/feishu_to_blog.py \
  --url 'https://dcnlcoue52zc.feishu.cn/wiki/SesqwwkKliHovjky4Ehc0WW2noh?from=from_copylink'
```

脚本会依次执行：

1. 用 `lark-cli drive +inspect` 解包 Wiki 链接并获取文档类型、标题和 token。
2. 用 `lark-cli docs +fetch --doc-format markdown` 读取正文。
3. 把 Feishu callout、grid、cite、whiteboard 等结构转成博客可渲染的 Markdown/HTML；分栏和 callout 图标、配色会保留。
4. 识别正文中的飞书图片 token，逐个用 `lark-cli docs +media-download` 下载到临时目录，全部成功后再安装到 `assets/images/`。
5. 生成 `notes/<slug>.md`，把所有图片改成本地相对路径。
6. 更新 `notes.json`，最后重新校验索引、图片、代码围栏和数学公式分隔符。

常用参数：

```bash
--slug lecture-03          # 覆盖自动 slug
--title 'Lecture 3: ...'   # 覆盖标题和 H1
--date 2026-10-20          # 默认今天
--description '摘要'        # 覆盖自动摘要
--overwrite                # 替换同 slug 的旧文章和旧图片
--repo /path/to/repo       # 不在仓库根目录运行时指定仓库
--json                     # 输出机器可读摘要
```

自动 slug 规则：标题含有 `Lecture 3`、`Lecture3` 或 `Lec 3` 时生成 `lecture-03`；否则从英文标题生成短横线 slug；纯中文标题回退到 `feishu-YYYYMMDD-<token 后缀>`。已有 slug 默认报错，避免误覆盖。

## 转换后验收

先看脚本摘要，再做仓库级校验：

```bash
python3 skills/feishu-to-blog/scripts/feishu_to_blog.py --check
git diff -- notes.json notes/<slug>.md
git status --short
```

逐项确认：

- `notes/<slug>.md` 只有一个 H1，且标题与 `notes.json` 一致。
- 所有 `![...](assets/images/...)` 路径都能在仓库中找到文件。
- 正文中没有 `https://feishu.cn/file/...` 形式的图片 URL。
- 正文中没有 `<callout>`、`<grid>`、`<whiteboard>`、`<cite>` 等未转换标签。
- 代码块使用成对围栏；公式使用 `$...$` 或 `$$...$$`，KaTeX 能处理。
- `notes.json` 没有重复 slug，日期是 `YYYY-MM-DD`。

如本机有静态服务器，可再打开 `note.html?p=<slug>` 目测，但不要为了验证而提交构建产物：

```bash
python3 -m http.server 8000
```

## 质量要求

转换后的 Markdown 必须兼容 `note.html` 当前的渲染链：

- 标题、列表、表格、引用使用标准 Markdown，不保留飞书专有 XML。
- 分栏使用 `<div class="feishu-grid">` / `<div class="feishu-column">`，栏宽来自 `width-ratio`，窄屏自动堆叠。
- 提示块使用 `<div class="feishu-callout">`，保留 emoji、背景色和边框色。
- 行内公式用 `$...$`，独立公式用 `$$...$$`；公式放在代码块外。
- 代码块保留语言标记，例如 ```python；不要转义围栏本身。
- 图片用 `assets/images/...` 相对路径；不要依赖飞书临时链接。
- 每个图片只下载一次；同一 token 多次出现时复用同一本地文件。
- 不修改与该文章无关的笔记、样式或索引。
- 用户没有明确要求时，不执行 `git commit`、`git push` 或部署。

如果自动摘要太泛，使用 `--overwrite --description '...'` 重跑，或直接改 `notes.json` 中该条目的 `description`，但不要破坏 JSON。

## 手动兜底

脚本失败时先定位阶段，再按同一顺序手工完成。详细规则见 `references/normalization.md`。

```bash
# 1. Wiki -> 底层文档
lark-cli drive +inspect --url "$URL" --as user --format json

# 2. 读取 Markdown 正文
lark-cli docs +fetch --doc "$URL" --doc-format markdown --detail simple --as user --format json

# 3. 下载正文中的图片 token
lark-cli docs +media-download --token "$TOKEN" \
  --output ./assets/images/<slug>-img-01 --overwrite --as user --format json
```

`lark-cli --output` 只接受相对当前工作目录的路径。图片 token 通常来自 Markdown 中的 `https://feishu.cn/file/<token>`。遇到 `permission_denied` 或 403 时，先尝试：

```bash
lark-cli docs +media-preview --token "$TOKEN" \
  --output ./assets/images/<slug>-img-01 --overwrite --as user --format json
```

如果仍失败，停止写入并报告具体图片 token；不要把失败的图片静默改成远程链接。外部非飞书图片会保留远程 URL 并在摘要中给出 warning。

## 失败处理

- `needs_refresh` / `no usable identity`：引导用户运行 `lark-cli auth login --domain docs,wiki,drive`，或在已授权环境重跑。
- 网络或 token 存储只读：请求联网/写权限后重跑；不要在沙箱中反复重试。
- `slug already exists`：确认是否更新旧文章；只有用户需要替换时才加 `--overwrite`。
- 图片下载失败：保留原文，不更新 `notes.json`；修复权限后用相同命令重跑。
- `validation failed`：按错误逐项修复；不要通过删除检查来让结果通过。
- 文档包含内嵌表格、多维表格或同步块：脚本转为飞书链接并给出 warning；若要求完整内容，先向用户说明限制，再按 `references/normalization.md` 扩展。

## 文件说明

- `scripts/feishu_to_blog.py`：转换、图片安装、索引更新与校验。
- `references/normalization.md`：飞书 Markdown 结构映射、渲染兼容性和排障细节。

