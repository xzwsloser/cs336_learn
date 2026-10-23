# CS336 Learning Notes

Stanford CS336: Language Modeling from Scratch 课程学习和作业笔记

## 在线阅读

https://xzwsloser.github.io/cs336_learn/

## 仓库内容

- 将课程笔记整理为可静态部署的 Markdown 页面，支持 LaTeX、代码语法高亮、图片和飞书分栏布局。
- 使用 `notes.json` 维护文章索引，`index.html` 渲染笔记列表，`note.html` 渲染单篇笔记。

页面渲染使用 marked、KaTeX 和 highlight.js，笔记源文件与图片均保存在仓库内，不依赖飞书临时资源。

## 仓库结构

```
├── index.html                  # 首页（笔记卡片列表）
├── note.html                   # 笔记渲染页（Markdown → HTML）
├── notes.json                  # 笔记索引（标题、日期、摘要、slug）
├── notes/                      # Markdown 笔记源文件
├── assets/
│   ├── css/style.css           # 全站样式
│   └── images/                 # 笔记中的本地图片
└── skills/
    └── feishu-to-blog/         # 飞书文档 → 博客笔记转换 skill
        ├── SKILL.md
        ├── references/
        │   └── normalization.md
        └── scripts/
            └── feishu_to_blog.py
```

## Skill：feishu-to-blog

`skills/feishu-to-blog` 是本仓库专用的文档转换 skill，用于把飞书/Lark Wiki 或 Docx 链接转换为一篇可直接发布的博客笔记。它面向当前静态站点约定，而不是通用的飞书导出工具。

主要作用：

- 通过 `lark-cli` 解包 Wiki 链接、读取文档正文，并保留标题、列表、表格、引用块、提示块和分栏结构。
- 将飞书图片逐个下载到 `assets/images/<slug>-img-NN.<ext>`，把正文中的临时链接替换为仓库内相对路径。
- 将飞书专有的 `grid`、`callout`、`cite`、`whiteboard` 等标签转换为 `note.html` 可渲染的 Markdown/HTML。
- 自动生成 `notes/<slug>.md`，并更新 `notes.json` 中的标题、日期和摘要。
- 转换后检查图片资源、代码围栏、LaTeX 分隔符、残留飞书标签和索引唯一性。

使用前需要安装并登录 `lark-cli`，且具备目标文档和图片的访问权限：

```bash
lark-cli whoami
python3 skills/feishu-to-blog/scripts/feishu_to_blog.py \
  --url 'https://example.feishu.cn/wiki/<wiki-token>' \
  --json
```

转换完成后执行仓库级校验：

```bash
python3 skills/feishu-to-blog/scripts/feishu_to_blog.py --check
```

脚本参数、失败处理和结构映射规则分别见 `SKILL.md` 与 `references/normalization.md`。嵌入表格、多维表格和同步块默认只保留链接，不会展开其数据；skill 也不会自动提交或推送 Git 变更。
