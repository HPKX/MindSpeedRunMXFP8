# DeepSeek V4 MXFP8 One-Page PPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一页可编辑的 DeepSeek V4 MXFP8 整体实施路线 PPT，并提供同版式 PNG 预览。

**Architecture:** 使用 `@oai/artifact-tool` 创建 1280×720 的单页演示文稿。页面采用固定坐标的“3 条工作线 × 6 个阶段”矩阵，Dense/MoE 依赖通过短竖向箭头嵌入对应阶段，工程可用卡片内呈现端到端验证与性能增强顺序。

**Tech Stack:** JavaScript ES Module、`@oai/artifact-tool`、PowerPoint `.pptx`、PNG 渲染、Presentation QA scripts。

## Global Constraints

- 最终文件：`outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.pptx`。
- 最终预览：`outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.png`。
- 画幅固定为 16:9，尺寸 1280×720。
- 页面使用三条工作线和六个阶段，不显示 A0、B0、C0 等内部编号。
- 第一阶段退出只检查基础算子与 Dense Runtime；网络线不作为退出条件。
- 性能增强顺序固定为：端到端验证 → 通算融合 → 梯度累加融合 → 缓存复用及其余优化 → 工程可用。
- 通算融合、梯度累加融合标注“算子 + Runtime”；缓存复用及其余优化标注“Runtime”。
- 生成脚本位于外部 scratch 目录，仓库只保留最终 PPT、PNG 和本实施计划。

---

### Task 1: 创建单页 PPT 生成脚本

**Files:**
- Create: `/private/tmp/codex-presentations/manual-20260718-mxfp8/mxfp8-one-page/tmp/build_mxfp8_roadmap.mjs`
- Create: `/Users/hz/Desktop/Code/MindSpeedRunMXFP8/outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.pptx`
- Create: `/Users/hz/Desktop/Code/MindSpeedRunMXFP8/outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.png`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-18-mxfp8-one-page-ppt-design.md` 中的六阶段内容、依赖关系和配色约束。
- Produces: `addHeader(slide)`、`addStageHeaders(slide, phases)`、`addDependencyArrows(slide)`、`addLaneGrid(slide, lanes)`、`addFooter(slide)` 和 `main(): Promise<void>`，导出 PPTX、PNG、layout JSON 和结构快照。

- [ ] **Step 1: 初始化 artifact-tool workspace**

Run:

```bash
mkdir -p /private/tmp/codex-presentations/manual-20260718-mxfp8/mxfp8-one-page/tmp
node "$SKILL_DIR/container_tools/setup_artifact_tool_workspace.mjs" \
  --workspace /private/tmp/codex-presentations/manual-20260718-mxfp8/mxfp8-one-page/tmp
```

Expected: scratch 目录出现可解析 `@oai/artifact-tool` 的 `package.json` 和依赖链接。

- [ ] **Step 2: 定义页面数据模型**

生成脚本中使用以下固定阶段和工作线数据：

```js
const phases = [
  ["阶段 1", "基础能力准备"],
  ["阶段 2 · 交付门", "Dense 网络汇聚"],
  ["阶段 3 · 交付门", "MoE 网络汇聚"],
  ["阶段 4 · 交付门", "工程可用"],
  ["阶段 5", "逐步上线"],
  ["阶段 6", "持续优化"],
];

const lanes = [
  { key: "op", title: "MindSpore\nMXFP8 算子", subtitle: "真实计算能力" },
  { key: "runtime", title: "MindSpeed\nMXFP8 Runtime", subtitle: "封装、调度与支撑" },
  { key: "network", title: "DeepSeek V4\n网络适配", subtitle: "项目汇聚主轴" },
];
```

Expected: 数据项与设计说明中的三条线、六阶段逐项一致。

- [ ] **Step 3: 实现固定布局和可编辑对象**

脚本使用以下入口与输出逻辑：

```js
import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const slide = deck.slides.add();
  slide.background.fill = "#F8FAFC";
  addHeader(slide);
  addStageHeaders(slide, phases);
  addDependencyArrows(slide);
  addLaneGrid(slide, lanes);
  addFooter(slide);
  const png = await deck.export({ slide, format: "png", scale: 2 });
  await writeBlob("/Users/hz/Desktop/Code/MindSpeedRunMXFP8/outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.png", png);
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile("/private/tmp/codex-presentations/manual-20260718-mxfp8/mxfp8-one-page/tmp/slide-1.layout.json", await layout.text());
  const snapshot = await deck.inspect({ kind: "slide,textbox,shape", maxChars: 30000 });
  await fs.writeFile("/private/tmp/codex-presentations/manual-20260718-mxfp8/mxfp8-one-page/tmp/slide-1.inspect.ndjson", snapshot.ndjson);
  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save("/Users/hz/Desktop/Code/MindSpeedRunMXFP8/outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.pptx");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

Expected: 文字、色块和箭头均为可编辑 PowerPoint 对象；无外部图片依赖。

- [ ] **Step 4: 运行生成脚本**

Run:

```bash
node /private/tmp/codex-presentations/manual-20260718-mxfp8/mxfp8-one-page/tmp/build_mxfp8_roadmap.mjs
```

Expected: 命令退出码为 0，PPTX、PNG、layout JSON 和 inspect NDJSON 全部生成。

### Task 2: 验证页面结构和视觉结果

**Files:**
- Test: `/Users/hz/Desktop/Code/MindSpeedRunMXFP8/outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.pptx`
- Test: `/Users/hz/Desktop/Code/MindSpeedRunMXFP8/outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.png`
- Test: `/private/tmp/codex-presentations/manual-20260718-mxfp8/mxfp8-one-page/tmp/slide-1.layout.json`

**Interfaces:**
- Consumes: Task 1 生成的 PPTX、PNG 和布局数据。
- Produces: 通过 `slides_test.py` 的画布边界检查，以及经过全尺寸目视检查的最终文件。

- [ ] **Step 1: 执行 PowerPoint 画布检查**

Run:

```bash
python "$SKILL_DIR/container_tools/slides_test.py" \
  /Users/hz/Desktop/Code/MindSpeedRunMXFP8/outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.pptx
```

Expected: 无超出 1280×720 画布的对象。

- [ ] **Step 2: 核对结构快照中的必要内容**

Run:

```bash
rg -n "MindSpore|MindSpeed|DeepSeek V4|Dense 网络汇聚|MoE 网络汇聚|通算融合|梯度累加融合|缓存复用|工程可用" \
  /private/tmp/codex-presentations/manual-20260718-mxfp8/mxfp8-one-page/tmp/slide-1.inspect.ndjson
```

Expected: 所有三条工作线、汇聚点和性能增强项均命中。

- [ ] **Step 3: 全尺寸检查 PNG**

打开 `outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.png`，逐项检查：

```text
标题单行显示；六个阶段标题无截断；三条工作线一眼可辨；
Dense/MoE 依赖箭头不穿过文本；阶段 4 顺序可读；
页脚阶段规则和项目主轴无重叠；所有卡片边界完整。
```

Expected: 无文字重叠、截断、异常换行和非预期遮挡。

- [ ] **Step 4: 若发现问题则修改并重新执行完整验证**

Run:

```bash
node /private/tmp/codex-presentations/manual-20260718-mxfp8/mxfp8-one-page/tmp/build_mxfp8_roadmap.mjs
python "$SKILL_DIR/container_tools/slides_test.py" \
  /Users/hz/Desktop/Code/MindSpeedRunMXFP8/outputs/DeepSeek_V4_MXFP8_Implementation_Roadmap.pptx
```

Expected: 修订后的生成和画布检查均退出码为 0，并重新完成 PNG 全尺寸检查。
