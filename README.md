# EntityIdentification
[![smithery badge](https://smithery.ai/badge/@u3588064/entityrecognition)](https://smithery.ai/server/@u3588064/entityrecognition)
Identify whether two sets of data are from the same entity. 识别两组数据是否来自同一主体

This is a MCP (Model Context Protocol) server. 这是一个支持MCP协议的服务器。


# Data Comparison Tool

This tool provides a comprehensive way to compare two sets of data, evaluating both exact and semantic equality of their values. It leverages text normalization and a language model to determine if the data originates from the same entity.

## Features

- **Text Normalization**: Converts text to lowercase, removes punctuation, and normalizes whitespace.
- **Value Comparison**: Compares values directly and semantically (ignoring order for lists).
- **JSON Traversal**: Iterates through each key in the JSON objects and compares corresponding values.
- **Language Model Integration**: Uses a generative language model to assess semantic similarity and provide a final judgment on whether the data comes from the same entity.

## Installation

### Installing via Smithery

To install Entity Identification for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@u3588064/entityrecognition):

```bash
npx -y @smithery/cli install @u3588064/entityrecognition --client claude
```

### Manual Installation
To use this tool, ensure you have the necessary dependencies installed. You can install them using pip:

```bash
pip install genai
```

## Usage

### Functions

1. **normalize_text(text)**:
   - Normalizes the input text by converting it to lowercase, removing punctuation, and normalizing whitespace.

2. **compare_values(val1, val2)**:
   - Compares two values both exactly and semantically.
   - If the values are lists, it ignores the order of elements for semantic comparison.

3. **compare_json(json1, json2)**:
   - Compares two JSON objects key by key.
   - Uses `compare_values` to evaluate each key's values.
   - Integrates a language model to assess semantic similarity and provides a final judgment.

### Example

```python
import json
import genai
import re

# Define your JSON objects
json1 = {
    "name": "John Doe",
    "address": "123 Main St, Anytown, USA",
    "hobbies": ["reading", "hiking", "coding"]
}

json2 = {
    "name": "john doe",
    "address": "123 Main Street, Anytown, USA",
    "hobbies": ["coding", "hiking", "reading"]
}

# Compare the JSON objects
comparison_results = compare_json(json1, json2)

# Generate final matching result
model1 = genai.GenerativeModel("gemini-2.0-flash-thinking-exp")
result_matching = model1.generate_content("综合这些信息，你认为可以判断两个数据来自同一主体吗？"+json.dumps(comparison_results, ensure_ascii=False, indent=4))
print(result_matching.text)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
## Contact

If you have any questions or suggestions, please contact me:
- Email: u3588064@connect.hku.hk
- GitHub: [u3588064@connect.hku.hk](mailto:u3588064@connect.hku.hk)。

Wechat
![qrcode_for_gh_643efb7db5bc_344(1)](https://github.com/u3588064/LLMemory/assets/53069671/8bb26c0f-4cab-438b-9f8c-16b1c26b3587)
