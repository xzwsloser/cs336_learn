# CS336 Learning Notes

Stanford CS336: Language Modeling from Scratch 课程学习笔记，使用 GitHub Pages 托管静态博客展示。

## 在线阅读

https://xzwsloser.github.io/cs336_learn/

## 仓库结构

```
├── index.html           # 首页（笔记卡片列表）
├── note.html            # 笔记渲染页（Markdown → HTML）
├── notes.json           # 笔记索引（标题、日期、摘要、slug）
├── notes/               # Markdown 源文件
│   └── lecture-01.md
└── assets/
    ├── css/style.css    # 全站样式
    └── images/          # 笔记中的本地图片
```

## 添加新笔记

1. 将飞书文档导出/转换为 Markdown，保存到 `notes/` 目录，命名如 `lecture-02.md`。
2. 如文档包含图片，将图片下载到 `assets/images/`，并在 Markdown 中使用相对路径 `assets/images/xxx.png`。
3. 在 `notes.json` 数组中新增一条记录：
   ```json
   {
     "slug": "lecture-02",
     "title": "Lecture 2: ...",
     "date": "2026-10-19",
     "description": "一句话摘要"
   }
   ```
4. 提交并推送到 `master` 分支，GitHub Pages 会自动重新部署。

## 本地预览

```bash
python3 -m http.server 8000
```

打开 http://localhost:8000 即可查看。
