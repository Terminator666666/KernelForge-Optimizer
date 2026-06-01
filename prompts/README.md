# Prompt Templates

本目录存储 KernelForge-Optimizer 的 prompt 模板。

模板是任务无关的。在开始 Agent 会话前，填入任务目标、约束、验证命令和晋升标准。

## Available Templates

| Path | Purpose |
|---|---|
| `kernel-optimization-flow.md` | CUDA kernel 优化任务的最小化 prompt |

## How To Use

1. 创建或进入任务实现工作空间
2. 将相关模板内容复制到 Agent 会话中
3. 用任务特定的详细信息替换占位符
4. 让 Agent 读取工作空间并写 `docs/draft.md`
5. 将草稿转换为可执行计划
6. 运行实现循环，每次有意义的变更后验证

任务特定的 prompts 应该与它们描述的任务放在一起。不要将特定 benchmark 的数据集、验收表或私有评估器详细信息添加到这个通用仓库。
