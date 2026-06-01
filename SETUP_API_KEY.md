# API Key 设置说明

## 问题

运行测试时出现错误：
```
Code generation failed: 请设置 DEEPSEEK_API_KEY 环境变量
```

## 解决方案

### 方法 1：设置环境变量（推荐）

在 Linux 环境中执行：

```bash
# 临时设置（仅当前终端会话有效）
export DEEPSEEK_API_KEY="your-api-key-here"

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export DEEPSEEK_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 方法 2：使用 .env 文件

1. 在项目根目录创建 `.env` 文件：
```bash
cd /path/to/KernelForge-Optimizer
nano .env
```

2. 添加以下内容：
```
DEEPSEEK_API_KEY=your-api-key-here
```

3. 修改代码加载 .env 文件（如果需要）

### 方法 3：使用其他 LLM 后端

如果没有 DeepSeek API key，可以使用其他 LLM：

#### OpenAI
```bash
export OPENAI_API_KEY="your-openai-key"
```

然后修改测试脚本：
```python
config = OptimizationConfig(
    benchmark_path=benchmark_path,
    max_iterations=5,
    llm_backend="openai",  # 改为 openai
    llm_model="gpt-4",     # 或 gpt-3.5-turbo
    output_dir="./test_results/matmul"
)
```

#### Anthropic (Claude)
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

```python
config = OptimizationConfig(
    benchmark_path=benchmark_path,
    max_iterations=5,
    llm_backend="anthropic",
    llm_model="claude-3-opus-20240229",
    output_dir="./test_results/matmul"
)
```

### 方法 4：跳过 LLM 测试（仅测试框架）

如果只想测试框架而不实际调用 LLM，可以修改代码使用 mock LLM。

## 获取 API Key

### DeepSeek
1. 访问 https://platform.deepseek.com/
2. 注册账号
3. 在 API Keys 页面创建新的 API key

### OpenAI
1. 访问 https://platform.openai.com/
2. 注册账号
3. 在 API Keys 页面创建新的 API key

### Anthropic
1. 访问 https://console.anthropic.com/
2. 注册账号
3. 在 API Keys 页面创建新的 API key

## 验证设置

运行以下命令验证 API key 是否设置成功：

```bash
# 检查环境变量
echo $DEEPSEEK_API_KEY

# 或者
env | grep API_KEY
```

## 重新运行测试

设置好 API key 后，重新运行测试：

```bash
python test_real_gpu.py
```

## 注意事项

1. **不要将 API key 提交到 Git**
   - 确保 `.env` 文件在 `.gitignore` 中
   - 不要在代码中硬编码 API key

2. **API 费用**
   - 使用 LLM API 会产生费用
   - 建议先用小规模测试（`max_iterations=1`）

3. **API 限流**
   - 注意 API 的速率限制
   - 如果遇到限流，可以增加重试延迟
