"""
LLM Query Interface - 适配 CudaForge 的 query_server 函数

提供简化的 query_model 接口，用于 KernelForge-Optimizer
"""

import os
import sys
from pathlib import Path
from typing import Optional

# 加载 .env 文件中的环境变量
def load_env_file():
    """从 .env 文件加载环境变量"""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue
                # 解析 KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # 只在环境变量不存在时设置
                    if key and not os.environ.get(key):
                        os.environ[key] = value

# 自动加载 .env 文件
load_env_file()

# 添加 CudaForge 到路径并导入其 query_server
cudaforge_query_server = None
CUDAFORGE_PATH = Path(__file__).parent.parent.parent / "CudaForge-main" / "CudaForge-main"

if CUDAFORGE_PATH.exists():
    # 将 CudaForge 路径添加到 sys.path
    cudaforge_path_str = str(CUDAFORGE_PATH)
    if cudaforge_path_str not in sys.path:
        sys.path.insert(0, cudaforge_path_str)

    try:
        # 动态导入 CudaForge 的 query_server 模块
        import importlib.util
        query_server_path = CUDAFORGE_PATH / "agents" / "query_server.py"
        spec = importlib.util.spec_from_file_location("cudaforge_query_server", query_server_path)
        cudaforge_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cudaforge_module)
        cudaforge_query_server = cudaforge_module.query_server
    except Exception as e:
        print(f"Warning: Failed to import CudaForge query_server: {e}")
        cudaforge_query_server = None
else:
    print(f"Warning: CudaForge not found at {CUDAFORGE_PATH}")


def query_model(
    prompt: str,
    backend: str = "deepseek",
    model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    system_prompt: str = "You are an expert CUDA kernel optimization assistant.",
    **kwargs
) -> str:
    """
    简化的 LLM 查询接口

    参数：
        prompt: 用户提示词
        backend: LLM 后端 ("deepseek", "openai", "anthropic", "google" 等)
        model: 模型名称
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        system_prompt: 系统提示词
        **kwargs: 其他参数传递给底层 query_server

    返回：
        str: LLM 生成的响应文本
    """
    # 如果 CudaForge 可用，使用其 query_server
    if cudaforge_query_server is not None:
        return cudaforge_query_server(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            server_type=backend,
            model_name=model,
            **kwargs
        )

    # 否则，使用本地实现（直接调用 API）
    return _query_model_direct(
        prompt=prompt,
        backend=backend,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt
    )


def _query_model_direct(
    prompt: str,
    backend: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str
) -> str:
    """
    直接调用 LLM API（当 CudaForge 不可用时）

    支持的后端：
    - ccvibe: CC-Vibe API (支持 Claude Opus 4.7 等多种模型)
    - deepseek: DeepSeek API
    - openai: OpenAI API
    - anthropic: Anthropic API
    """
    if backend == "ccvibe":
        return _query_ccvibe(prompt, model, temperature, max_tokens, system_prompt)
    elif backend == "deepseek":
        return _query_deepseek(prompt, model, temperature, max_tokens, system_prompt)
    elif backend == "openai":
        return _query_openai(prompt, model, temperature, max_tokens, system_prompt)
    elif backend == "anthropic":
        return _query_anthropic(prompt, model, temperature, max_tokens, system_prompt)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _query_ccvibe(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str
) -> str:
    """调用 CC-Vibe API (支持 Claude Opus 4.7 等多种模型)"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai 包: pip install openai")

    api_key = os.environ.get("CCVIBE_API_KEY")
    base_url = os.environ.get("CCVIBE_BASE_URL", "https://cc-vibe.com/v1")

    if not api_key:
        raise ValueError("请设置 CCVIBE_API_KEY 环境变量")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=10000000,
        max_retries=3,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 打印使用统计
    if hasattr(response, 'usage') and response.usage:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) if hasattr(response.usage, 'prompt_tokens_details') else 0
        print(f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}, Cached={cached_tokens}")

    return response.choices[0].message.content


def _query_deepseek(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str
) -> str:
    """调用 DeepSeek API"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai 包: pip install openai")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=10000000,
        max_retries=3,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 打印使用统计
    if hasattr(response, 'usage') and response.usage:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        print(f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}")

    return response.choices[0].message.content


def _query_openai(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str
) -> str:
    """调用 OpenAI API"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai 包: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 打印使用统计
    if hasattr(response, 'usage') and response.usage:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        print(f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}")

    return response.choices[0].message.content


def _query_anthropic(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str
) -> str:
    """调用 Anthropic API"""
    try:
        import anthropic
    except ImportError:
        raise ImportError("请安装 anthropic 包: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("请设置 ANTHROPIC_API_KEY 环境变量")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 打印使用统计
    if hasattr(response, 'usage'):
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        print(f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}")

    # 提取文本内容
    outputs = []
    for block in response.content:
        if hasattr(block, "text"):
            outputs.append(block.text)

    return outputs[0] if outputs else ""


# 向后兼容：导出 query_server 别名
query_server = query_model
