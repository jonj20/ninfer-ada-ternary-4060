# AGENTS.md — Python 项目

## 项目概述
- Python 版本：3.12+
- 依赖管理：uv（禁止 uv pip install，使用 uv add / uv sync）
- 核心目标：确保 AI 生成的代码注释与日志符合团队规范，中文优先、简洁、结构化。

## 编码规范（所有任务必须遵守）

### 注释
- **语言**：中文优先，英文辅助。docstring 和行内注释均使用中文。
- **格式**：所有公共函数/类/模块必须包含 docstring，采用 **Google 风格**（Args / Returns / Raises）。
- **行内注释**：`#` 注释必须独立成行，禁止写在代码行末尾。
- **内容**：只解释“为什么”和“非显而易见的意图”，绝不叙述代码“做了什么”。
- **禁止事项**：
  - 禁止英文注释（代码标识符和 API 名称除外）。
  - 禁止叙述式注释（`# 这里我们...`、`# 然后...`、`# 首先...`）。
  - 禁止在注释中写 PR/Issue 编号或 GitHub URL。
  - 禁止注释掉的代码。
  - 禁止口语词（跑、起、关掉、拿不到、还活着）。
  - 禁止 emoji。
  - 禁止成功横幅（`✅ ...`）。

### Docstring 规范（Google 风格）
- 模块、类、公开函数必须包含 docstring。
- 函数 docstring 必须包含 `Args`、`Returns`、`Raises` 三段（无参数/无返回值时省略对应段）。
- 参数描述必须与函数签名一致，禁止占位描述（如 `param x: x 值`）。
- 禁止使用 `@param` / `@return` 等遗留 JSDoc 标签。
- 复杂函数应包含可运行的 `Example` 段。

### 日志
- **禁止**：在生产代码中使用 `print()`，包括调试输出。
- `print()` 仅允许两种场景：CLI 脚本的最终用户可见输出；测试中断言之外的诊断输出（推荐用 logger）。
- **要求**：使用标准库 `logging` 模块，或项目统一日志库（`loguru` / `structlog`）。
- **结构化**：动态值通过 `extra` 参数或 `%s` 占位符传递，**禁止 f-string 预格式化**。
- **消息格式**：静态字符串，末尾不加句号，不包含动态值插值。
- 日志只承担三类输出：观测（`<主体>: <字段>=<值>`）、跳过（`跳过: <原因>`）、构造（`构造: <字段>=<值>`）。

### 类型与代码风格
- 使用类型提示（type hints），所有公开函数必须有完整签名。
- 优先使用 `pathlib` 而非 `os.path`。
- 使用 `dataclasses` 或 `pydantic` 定义数据模型。
- 禁止裸 `except`，必须捕获具体异常或 `Exception` 并记录日志。

## CI/CD 检查命令
- Lint：`ruff check .`
- 格式：`ruff format --check .`
- 类型：`mypy .`
- 日志检查：由ruff拓展规则代劳
- 测试：`pytest`

## 可用技能（Available Skills）

### 注释生成与规范
- **`docstring-generator`**：按 Google/NumPy 风格生成 docstring，支持 Python。
- **`code-documenter`**：自动检测语言并应用对应文档标准。

### 注释清理与后处理
- **`code-polish`**：将非专业注释重写为清晰、专业的表述，不改变代码逻辑。
- **`remove-llm-comments`**：移除冗余、叙述性的 LLM 注释。

### 日志约束
- **`logging-best-practices`**：仓库感知的日志规范执行，含 ruff / mypy / pytest 验证。
- **`logging-guidelines`**：禁止 print、规范 logger 使用、日志级别划分。

### 中文优化
- **`chinese-ai-coding-skills`**：中文注释场景专项优化，防止 AI 删除中文注释或替换为英文。
