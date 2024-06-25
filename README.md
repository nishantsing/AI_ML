# AI_ML

## Integrating your local large language model to VSCode
- [Ollama - Download](https://ollama.com/)
- ollama --help in terminal to check if its installed
- [To check Models ranking](https://evalplus.github.io/leaderboard.html)
- go to ollama website and search for CodeQwen(4GB) or DeepSeek code v2(8GB) models because they are free unlike GPT-4
- ollama run codeqwen
- continue(own AI copilot) extension in VSCode
- create a customCommands
```json
{
  "name":"step",
  "prompt":"{{{input}}}\n\nExplain the selected code step by step",
  "description":"Code explanation"
}

```
- Ctrl + I -> ask the model to do generate some code
- "Select the code "Ctrl + L
- 
