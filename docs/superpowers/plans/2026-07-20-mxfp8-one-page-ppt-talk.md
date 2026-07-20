# DeepSeek V4 MXFP8 One-Page PPT Talk Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 编写一份面向技术方案汇报、时长约 8 分钟、可直接照读的 DeepSeek V4 MXFP8 单页 PPT 中文讲演稿。

**Architecture:** 讲演稿采用“先横后纵”的结构：先讲三条工作线，再沿六阶段解释交付依赖，重点展开工程可用阶段，最后回到排期结论。正文按时间段组织，每段包含指图提示、照读正文和强调词，并通过字符数、主题覆盖和一致性检查验收。

**Tech Stack:** Markdown、shell 文本检查、Git。

## Global Constraints

- 设计依据：`docs/superpowers/specs/2026-07-18-mxfp8-one-page-ppt-talk-design.md`。
- 对应页面：`outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.pptx` 第 1 页。
- 最终讲演稿：`docs/DeepSeek_V4_MXFP8_Presentation_Script.md`。
- 汇报类型：技术方案汇报。
- 正文时长：约 8 分钟。
- 正文长度：约 1800–2200 个中文字符；允许因英文术语和指图过渡扩展至 2400 字符。
- 核心结构：总体结论 → 三线职责 → 阶段一 → Dense/MoE 交付门 → 工程可用 → 上线与持续优化 → 排期收口。
- 不展开具体接口签名、算子实现或网络代码修改点。
- 通算融合和梯度累加融合归属“MindSpore 算子 + MindSpeed Runtime”；缓存复用及其余优化归属“MindSpeed Runtime”。

---

### Task 1: 编写正式讲演稿

**Files:**
- Create: `docs/DeepSeek_V4_MXFP8_Presentation_Script.md`
- Read: `docs/superpowers/specs/2026-07-18-mxfp8-one-page-ppt-talk-design.md`
- Read: `docs/superpowers/specs/2026-07-18-mxfp8-one-page-ppt-design.md`

**Interfaces:**
- Consumes: 讲演稿设计说明中的时间分配、叙事顺序、责任边界和验收口径。
- Produces: 包含 `<!-- SPEECH START -->` 与 `<!-- SPEECH END -->` 标记的正式照读正文，以及每段的指图提示和强调词。

- [ ] **Step 1: 建立讲演稿固定结构**

创建文件并使用以下一级、二级标题：

```markdown
# DeepSeek V4 MXFP8 整体实施路线讲演稿

> 场景：技术方案汇报
>
> 时长：约 8 分钟
>
> 对应页面：DeepSeek V4 MXFP8 整体实施路线

<!-- SPEECH START -->

## 0:00–0:45｜开场与总体结论
## 0:45–2:15｜三条工作线及职责边界
## 2:15–3:05｜阶段一：基础能力准备
## 3:05–4:25｜Dense 与 MoE 两个交付门
## 4:25–6:30｜工程可用：验证与性能增强闭环
## 6:30–7:20｜逐步上线与持续优化
## 7:20–8:00｜排期结论与收口

<!-- SPEECH END -->

## 讲演提醒
```

Expected: 七个时间段与设计说明的时间分配完全一致。

- [ ] **Step 2: 编写开场和三线职责正文**

开场必须包含以下完整结论：

```text
三条工作线并行推进；实际可运行能力按照“算子 → Runtime → 网络”逐层交付；项目级汇聚和验收以网络结果为主轴。
```

三线职责必须分别说明：

```text
MindSpore 算子：真实 MXFP8 计算能力。
MindSpeed Runtime：契约、上下文、调度和工程支撑。
DeepSeek V4 网络：BF16 基线、实际覆盖范围、配置启停和端到端验收。
```

Expected: 听众在前 2 分钟能够理解三条线的边界和上下游关系。

- [ ] **Step 3: 编写阶段一和 Dense/MoE 交付门正文**

阶段一必须说明：

```text
网络线正常并行推进，但不作为阶段一退出条件；阶段一只检查基础算子和 Dense Runtime 是否达到网络接入条件。
```

Dense/MoE 交付门必须按照以下顺序讲述：

```text
基础算子 → Dense Runtime → Dense 网络汇聚
MoE 算子 → MoE Runtime → MoE 网络汇聚
```

Expected: 明确算子早于 Runtime 验证窗口、Runtime 早于网络联调窗口的排期关系，同时说明 Dense 汇聚期间提前推进 MoE。

- [ ] **Step 4: 编写工程可用、上线和收口正文**

工程可用部分必须按照以下顺序展开：

```text
端到端验证与瓶颈定位
→ 通算融合
→ 梯度累加融合
→ 缓存复用及其余优化
→ 端到端复验
→ 工程可用
```

上线和收口必须说明：

```text
从验证任务开始逐步扩大任务和训练规模；异常时关闭对应低精配置并执行 BF16 路径；项目整体以 Dense 汇聚、MoE 汇聚、工程可用和逐步上线四个网络节点衡量。
```

Expected: 工程可用部分占全文最大篇幅，责任归属和端到端验收口径清晰。

- [ ] **Step 5: 添加指图提示、强调词和讲演提醒**

每个时间段在正文前增加一行：

```markdown
**指图：** 页面左侧三条工作线
```

七段分别使用以下指图位置：整体页面、左侧三条线、阶段一、Dense/MoE 竖向箭头、阶段四橙色卡片、阶段五与阶段六、底部项目主轴。

每段末尾增加一行，并按顺序使用以下强调词：总体框架；职责边界；首阶段退出；逐层交付；验证与性能增强；配置启停；网络主轴。

```markdown
**强调：** 三线并行、逐层交付、网络主轴
```

文件末尾的“讲演提醒”包含：控制语速、避免逐卡朗读、阶段四适当停顿、结尾回到排期结论。

Expected: 讲演者可以不依赖额外备注完成指图和节奏控制。

### Task 2: 验证并提交讲演稿

**Files:**
- Test: `docs/DeepSeek_V4_MXFP8_Presentation_Script.md`
- Test: `docs/superpowers/specs/2026-07-18-mxfp8-one-page-ppt-talk-design.md`

**Interfaces:**
- Consumes: Task 1 生成的 Markdown 讲演稿。
- Produces: 通过占位符、内容覆盖、正文长度和 Git 差异检查的正式稿。

- [ ] **Step 1: 检查占位符和空白错误**

Run:

```bash
if rg -n 'TB''D|TO''DO|待''补充|稍后''填写' docs/DeepSeek_V4_MXFP8_Presentation_Script.md; then exit 1; fi
if rg -n ' +$' docs/DeepSeek_V4_MXFP8_Presentation_Script.md; then exit 1; fi
```

Expected: 两个检查均退出码为 0 且无命中。

- [ ] **Step 2: 检查关键内容覆盖**

Run:

```bash
rg -n "三条工作线|算子 → Runtime → 网络|BF16 基线|第一阶段|Dense Runtime|MoE Runtime|通算融合|梯度累加融合|缓存复用|逐步上线|关闭.*低精|工程可用" docs/DeepSeek_V4_MXFP8_Presentation_Script.md
```

Expected: 每个关键主题至少命中一次，性能增强顺序和责任归属完整。

- [ ] **Step 3: 检查正文长度**

Run:

```bash
sed -n '/<!-- SPEECH START -->/,/<!-- SPEECH END -->/p' docs/DeepSeek_V4_MXFP8_Presentation_Script.md | wc -m
```

Expected: 输出位于 1800–2400 字符之间。

- [ ] **Step 4: 对照设计说明复核时间段和叙事顺序**

Run:

```bash
rg -n '^## [0-9]:[0-9][0-9]–[0-9]:[0-9][0-9]' docs/DeepSeek_V4_MXFP8_Presentation_Script.md
```

Expected: 七个时间段按 0:00–0:45 到 7:20–8:00 的顺序出现。

- [ ] **Step 5: 提交正式讲演稿**

Run:

```bash
git add docs/DeepSeek_V4_MXFP8_Presentation_Script.md
git diff --cached --check
git commit -m "docs: add MXFP8 presentation script"
```

Expected: 提交只包含正式讲演稿，不包含用户现有的其他工作区改动。
