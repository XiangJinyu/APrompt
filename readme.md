# APrompt 🤖

![Project Banner](readme_files/banner.png)

An automated prompt engineering tool for Large Language Models (LLMs), designed for universal domain adaptation.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## ✨ Features

- 💰 Ultra-low iteration cost ($0.1 level)
- 🏷️ Label-free approach - no training data needed
- 📊 Few-shot learning with minimal examples
-  ⚡ Super simple template configuration
- 🌐 Multi-language support

## 🚀 Quick Start

### 1. Configure Your API Key ⚙️

Create a configuration file `config.yaml`:

```yaml
openai:
  api_key: " "
  base_url: " "
```

### 2. Define Your Prompt Template 📝

Create a prompt template file `settings/task_name.yaml`:
```yaml
prompt: |
  solve question.

requirements: |
  ...

count: None

faq:
  - question: |
      ...
    answer: |
      ...

  - question: |
      ...
    answer: |
      ...
```

### 3. Implement the Optimizer 🔧

Use `main.py` to execute:
```python
from script.optimizer import Optimizer

if __name__ == "__main__":

    optimizer = Optimizer(
        optimized_path="workspace",
        initial_round=1,
        max_rounds=10,
        template="task_name.yaml",
        name="Task",
        optimize_model={"name": "claude-3-5-sonnet-20240620", "temperature": 0.7},
        execute_model={"name": "gpt-4o-mini", "temperature": 0},
        evaluate_model={"name": "gpt-4o-mini", "temperature": 0.3},
        iteration=True,
    )

    optimizer.optimize()
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ by the MetaGPT and AFlow Team</p>