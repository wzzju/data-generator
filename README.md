# Data Generator

一个基于 Rust 编写的 CLI 工具，用于从文本语料中随机截取文本，生成指定格式的数据集。支持通过 HuggingFace Tokenizers 精准控制 Token 数量。

## 功能特性

- 从多个 txt 语料文件中随机截取文本
- 支持生成两种格式的数据集：
  - **AIAK 格式**：JSON 格式，包含多轮 human-gpt 对话
  - **Bench 格式**：JSONL 格式，每行一个 prompt
- 支持指定 Token 数量范围（最小值、最大值、平均值）
- 使用 [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) 精准计算 Token 数量
- 截取文本时自动对齐到句子边界，确保开头为完整句子

## 环境准备

### 安装 Rust 编译环境

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

安装完成后，重新加载环境变量：

```bash
source $HOME/.cargo/env
```

验证安装：

```bash
rustc --version
cargo --version
```

> **注意**：本项目使用 Rust Edition 2024，请确保 Rust 工具链版本 ≥ 1.85.0。可通过 `rustup update` 更新到最新版本。

## 编译

### Debug 编译（用于开发调试）

```bash
cargo build
```

编译产物位于 `target/debug/data-generator`。

### Release 编译（用于生产环境，开启优化）

```bash
cargo build --release
```

编译产物位于 `target/release/data-generator`。

### 运行测试

```bash
cargo test
```

## 使用方法

### 基本语法

```bash
data-generator -i <输入文件>... -f <格式> -o <输出文件> -t <词表文件> -r <Token范围> [-c <条目数>]
```

### 参数说明

| 参数 | 缩写 | 必填 | 说明 |
|------|------|------|------|
| `--input` | `-i` | ✅ | 输入语料文件路径，至少一个，可指定多个 |
| `--format` | `-f` | ✅ | 输出格式，可选 `aiak` 或 `bench` |
| `--output` | `-o` | ✅ | 输出文件路径 |
| `--tokenizer` | `-t` | ✅ | HuggingFace tokenizer.json 词表文件路径 |
| `--token-range` | `-r` | ✅ | Token 数量范围（见下方语法说明） |
| `--count` | `-c` | ❌ | 生成的条目数量，默认为 10 |

### Token 范围语法

Token 范围使用 `[min-]max[:avg]` 格式指定：

| 格式 | 含义 |
|------|------|
| `300` | max=300（min 和 avg 自动推算） |
| `100-300` | min=100, max=300 |
| `100-300:200` | min=100, max=300, avg=200 |
| `300:200` | max=300, avg=200 |

- **min**（最小值）：省略时默认为 max 的 1/4
- **avg**（平均值）：省略时默认为 (min + max) / 2
- **max**（最大值）：必须指定

### 使用示例

#### 生成 Bench 格式数据集

```bash
cargo run --release -- \
  -i demo/input_0.txt \
  -i demo/input_1.txt \
  -f bench \
  -o output_bench.jsonl \
  -t demo/tokenizer.json \
  -r "100-300:200" \
  -c 20
```

#### 生成 AIAK 格式数据集

```bash
cargo run --release -- \
  -i demo/input_0.txt \
  -i demo/input_1.txt \
  -f aiak \
  -o output_aiak.json \
  -t demo/tokenizer.json \
  -r "50-200:100" \
  -c 15
```

#### 仅指定 max Token 数

```bash
cargo run -- \
  -i demo/input_0.txt \
  -f bench \
  -o output.jsonl \
  -t demo/tokenizer.json \
  -r 500
```

## 输出格式示例

### AIAK 格式（JSON）

```json
[
  {
    "id": "i6IyJda_0",
    "conversations": [
      {
        "from": "human",
        "value": "截取的语料文本内容..."
      },
      {
        "from": "gpt",
        "value": "紧随 human 后文的语料文本内容..."
      }
    ]
  }
]
```

每条数据包含：
- **id**：唯一标识符
- **conversations**：多轮对话列表，human 和 gpt 成对出现（至少一对）
- 每个 id 对应的多轮 human-gpt 对话内容 Token 总和在用户指定的范围内

### Bench 格式（JSONL）

```jsonl
{"prompt":"截取的语料文本内容..."}
{"prompt":"截取的语料文本内容..."}
```

每行一个 JSON 对象，包含一个 `prompt` 字段。

## 项目结构

```
data-generator/
├── Cargo.toml          # 项目配置和依赖管理
├── Cargo.lock          # 依赖锁定文件
├── src/
│   ├── main.rs         # 程序入口
│   ├── cmd.rs          # CLI 参数定义
│   ├── error.rs        # 错误处理
│   ├── generator.rs    # 数据集生成核心逻辑
│   └── token_range.rs  # Token 范围解析
└── demo/
    ├── tokenizer.json  # 示例词表文件
    ├── input_0.txt     # 示例输入语料
    ├── inpu_1.txt      # 示例输入语料
    ├── aiak_dataset.json   # AIAK 格式示例输出
    └── bench_dataset.jsonl # Bench 格式示例输出
```

## License

MIT
