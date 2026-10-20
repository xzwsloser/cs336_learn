# Feishu Markdown Normalization

本文件说明 `lark-cli docs +fetch --doc-format markdown` 的常见输出形态，以及本仓库 `note.html` 的渲染约束。只有在需要修改脚本、处理新标签或排查转换质量时读取。

## 1. 飞书输出与本仓库渲染链

飞书 Markdown 不是 CommonMark 的严格子集，常见残留包括：

- `<callout emoji="...">...</callout>`：提示块。
- `<grid>` / `<column width-ratio="...">`：分栏布局。
- `<cite doc-id="..." file-type="..." title="..."></cite>`：文档引用。
- `<whiteboard token="..."/>`：画板。
- `<source token="..." name="..."/>`：附件。
- `<sheet token="..."/>`、`<bitable token="..."/>`：嵌入表格。
- `<blockquote><p>...</p></blockquote>`：HTML 引用块。
- 标题中带 `**粗体**`，例如 `## **随笔**`。

`note.html` 的渲染链是 marked（GFM）、KaTeX 自动渲染、highlight.js。marked 会保留部分 HTML，但飞书专有标签不会被浏览器正确展示，因此转换时必须归一化。

## 2. 标签映射

| 飞书结构 | 目标 Markdown | 说明 |
| --- | --- | --- |
| `<callout emoji="🤔">` | `> 🤔 ...` | 每行加 `>`；保留换行；不保留 callout 背景色 |
| `<grid>` / `<column>` | 删除包裹标签 | 分栏改为顺序图片/段落，避免内容丢失 |
| `<cite>` | `[title](https://feishu.cn/...)` | 按 `file-type` 选择 wiki/docx/sheets/base 路径 |
| `<whiteboard token>` | `![飞书画板](assets/images/...)` | 下载画板缩略图 |
| `<source token name>` | `[📎 name](https://feishu.cn/file/token)` | 附件保留链接，不承诺本地离线 |
| `<sheet token>` | `[📊 飞书表格](https://feishu.cn/sheets/token)` | 表格数据不展开 |
| `<bitable token>` | `[📊 飞书多维表格](https://feishu.cn/base/token)` | Base 数据不展开 |
| `<synced_reference>` | `🔗 标题` 链接 | 同步块内容不递归展开 |
| `<blockquote><p>` | `> ...` | 删除包裹 HTML，保留引用语义 |
| `<p>` | 普通段落 | 避免 HTML 块吞掉内部 Markdown 链接 |
| `## **标题**` | `## 标题` | 去掉标题内多余的粗体标记 |

## 3. 图片处理

飞书 Markdown 通常把图片写成：

```markdown
![图片描述](https://feishu.cn/file/<file_token>)
```

处理流程：

1. 从 `/file/<token>`、`/media/<token>` 或下载路径中提取 token。
2. 同一 token 只下载一次。
3. 使用 `lark-cli docs +media-download --token <token> --output ./.feishu-to-blog-tmp/<slug>-img-01 --overwrite --as user --format json`。
4. 根据 `saved_path` 的扩展名或 `content_type` 决定最终扩展名。
5. 全部下载成功后再移动到 `assets/images/<slug>-img-NN.<ext>`。
6. 将 Markdown 改写为 `![alt](assets/images/<slug>-img-NN.<ext>)`。

注意事项：

- `--output` 必须是当前工作目录内的相对路径；脚本通过 `cwd=<repo>` 满足这个约束。
- 下载失败时不要更新 `notes.json`，否则首页会出现打不开的图片。
- 403 / permission denied 时尝试 `docs +media-preview`。
- 外部 `https://` 图片没有飞书 token 时保留远程 URL，并在摘要中 warning。
- 不要把图片存到 `notes/` 内；本仓库页面从根目录解析 `assets/images/...`。

## 4. 数学公式

飞书通常输出 `$...$` 和 `$$...$$`。保留这些定界符，不要转成 `\(...\)`、`\[...\]` 或 HTML。

检查时先从正文中排除代码围栏和行内代码，再检查：

- `$$` 数量为偶数。
- 未被转义的 `$` 数量为偶数；不满足时给 warning，而不是自动改写金额或普通文本中的美元符号。
- 公式内部的反斜杠保持原样，例如 `\frac`、`\mathbf`、`\Theta`。
- 不让公式跨入飞书 callout 的行内代码或代码块。

`note.html` 已使用 KaTeX auto-render，并忽略 `pre` / `code`，因此代码中的 `$` 不会被误渲染。

## 5. 代码块与表格

- 代码围栏必须成对，语言标记保留。
- callout 内出现代码时，每行仍以 `> ` 开头，KaTeX 会忽略 `pre/code`。
- 飞书表格若已输出标准 Markdown 管道表，直接保留；分隔行必须包含 `---`。
- 不把 `<sheet>` / `<bitable>` 当作普通 Markdown 表格强行展开，除非用户明确需要数据同步。

## 6. 索引与 slug

`notes.json` 的每条记录必须包含：

```json
{
  "slug": "lecture-03",
  "title": "Lecture 3: Model Architecture",
  "date": "2026-10-20",
  "description": "一句话摘要"
}
```

规则：

- `slug` 只允许小写字母、数字和连字符。
- `notes/<slug>.md` 与 `notes.json` 条目一一对应。
- 标题含 `Lecture N` 时优先使用 `lecture-0N`。
- 更新同一文章必须显式使用 `--overwrite`，脚本会删除该 slug 的旧 `<slug>-img-*` 图片。
- `title` 和 `description` 只放纯文本；首页会用 innerHTML 拼接，脚本会做 HTML 转义。

## 7. 验收清单

```bash
python3 skills/feishu-to-blog/scripts/feishu_to_blog.py --check
rg -n 'feishu\.cn/file/|<callout|<grid|<whiteboard|<cite' notes/<slug>.md
rg -n '^```|^\$\$' notes/<slug>.md
```

预期：

- 第一条 `rg` 无命中。
- 代码围栏成对，公式定界符成对。
- 所有本地图片存在。
- `notes.json` 仍为合法 JSON，且 slug 唯一。

## 8. 已知限制

- 飞书评论不会进入博客正文。
- 画板只保留缩略图，不保留可编辑源。
- 分栏布局线性化，不还原原始并排宽度。
- 嵌入表格、多维表格和同步块转为链接，不复制其数据。
- 外部图片不做 SSRF 风险下载，只保留远程 URL 并 warning。
