<p align="center">
  <a href="https://github.com/mem0ai/mem0">
    <img src="docs/images/banner-sm.png" width="800px" alt="Mem0 - The Memory Layer for Personalized AI">
  </a>
</p>
<p align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <a href="https://trendshift.io/repositories/11194" target="blank">
    <img src="https://trendshift.io/api/badge/repositories/11194" alt="mem0ai%2Fmem0 | Trendshift" width="250" height="55"/>
  </a>
</p>

<p align="center">
  <a href="https://mem0.ai">Learn more</a>
  ·
  <a href="https://mem0.dev/DiG">Join Discord</a>
  ·
  <a href="https://mem0.dev/demo">Demo</a>
</p>

<p align="center">
  <a href="https://mem0.dev/DiG">
    <img src="https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white" alt="Mem0 Discord">
  </a>
  <a href="https://pepy.tech/project/mem0ai">
    <img src="https://img.shields.io/pypi/dm/mem0ai" alt="Mem0 PyPI - Downloads">
  </a>
  <a href="https://github.com/mem0ai/mem0">
    <img src="https://img.shields.io/github/commit-activity/m/mem0ai/mem0?style=flat-square" alt="GitHub commit activity">
  </a>
  <a href="https://pypi.org/project/mem0ai" target="blank">
    <img src="https://img.shields.io/pypi/v/mem0ai?color=%2334D058&label=pypi%20package" alt="Package version">
  </a>
  <a href="https://www.npmjs.com/package/mem0ai" target="blank">
    <img src="https://img.shields.io/npm/v/mem0ai" alt="Npm package">
  </a>
  <a href="https://www.ycombinator.com/companies/mem0">
    <img src="https://img.shields.io/badge/Y%20Combinator-S24-orange?style=flat-square" alt="Y Combinator S24">
  </a>
</p>

<p align="center">
  <a href="https://mem0.ai/research"><strong>📄 Building Production-Ready AI Agents with Scalable Long-Term Memory →</strong></a>
</p>
<p align="center">
  <strong>⚡ +26% Accuracy vs. OpenAI Memory • 🚀 91% Faster • 💰 90% Fewer Tokens</strong>
</p>

> **🎉 mem0ai v1.0.0 is now available!** This major release includes API modernization, improved vector store support, and enhanced GCP integration. [See migration guide →](MIGRATION_GUIDE_v1.0.md)

##  🔥 Research Highlights
- **+26% Accuracy** over OpenAI Memory on the LOCOMO benchmark
- **91% Faster Responses** than full-context, ensuring low-latency at scale
- **90% Lower Token Usage** than full-context, cutting costs without compromise
- [Read the full paper](https://mem0.ai/research)

# Introduction

[Mem0](https://mem0.ai) ("mem-zero") enhances AI assistants and agents with an intelligent memory layer, enabling personalized AI interactions. It remembers user preferences, adapts to individual needs, and continuously learns over time—ideal for customer support chatbots, AI assistants, and autonomous systems.

### Key Features & Use Cases

**Core Capabilities:**
- **Multi-Level Memory**: Seamlessly retains User, Session, and Agent state with adaptive personalization
- **Developer-Friendly**: Intuitive API, cross-platform SDKs, and a fully managed service option

**Applications:**
- **AI Assistants**: Consistent, context-rich conversations
- **Customer Support**: Recall past tickets and user history for tailored help
- **Healthcare**: Track patient preferences and history for personalized care
- **Productivity & Gaming**: Adaptive workflows and environments based on user behavior

## 🚀 Quickstart Guide <a name="quickstart"></a>

Choose between our hosted platform or self-hosted package:

### Hosted Platform

Get up and running in minutes with automatic updates, analytics, and enterprise security.

1. Sign up on [Mem0 Platform](https://app.mem0.ai)
2. Embed the memory layer via SDK or API keys

### Self-Hosted (Open Source)

Install the sdk via pip:

```bash
pip install mem0ai
```

Install sdk via npm:
```bash
npm install mem0ai
```

### Basic Usage

Mem0 requires an LLM to function, with `gpt-4.1-nano-2025-04-14 from OpenAI as the default. However, it supports a variety of LLMs; for details, refer to our [Supported LLMs documentation](https://docs.mem0.ai/components/llms/overview).

First step is to instantiate the memory:

```python
from openai import OpenAI
from mem0 import Memory

openai_client = OpenAI()
memory = Memory()

def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # Retrieve relevant memories
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])

    # Generate Assistant response
    system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    response = openai_client.chat.completions.create(model="gpt-4.1-nano-2025-04-14", messages=messages)
    assistant_response = response.choices[0].message.content

    # Create new memories from the conversation
    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)

    return assistant_response

def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")

if __name__ == "__main__":
    main()
```

For detailed integration steps, see the [Quickstart](https://docs.mem0.ai/quickstart) and [API Reference](https://docs.mem0.ai/api-reference).

---

## ☁️ Azure AI Foundry Setup (sia-consulting fork)

This fork adds an **Azure AI Foundry** provider for both LLM and embeddings, powered by the [`azure-ai-inference`](https://pypi.org/project/azure-ai-inference/) SDK. It works with **any model** deployed to Azure AI Foundry — including OpenAI (GPT-4o, GPT-4.1), Anthropic (Claude), Meta (Llama), Mistral, and more.

### Installation

Install directly from the fork's GitHub repository:

```bash
pip install "mem0ai @ git+https://github.com/sia-consulting/mem0.git"
```

Or install a specific release tag (e.g. `v1.0.8`):

```bash
pip install "mem0ai @ git+https://github.com/sia-consulting/mem0.git@v1.0.8"
```

Or install the extra dependencies for Azure AI Foundry support on any mem0 install:

```bash
pip install "azure-ai-inference>=1.0.0b9" "azure-identity>=1.24.0"
```

### Prerequisites

1. **Azure AI Foundry resource** — create one in the [Azure Portal](https://portal.azure.com) or via the [Azure AI Foundry portal](https://ai.azure.com).
2. **Model deployments** — deploy a chat completion model (e.g. `gpt-4o`) and an embedding model (e.g. `text-embedding-3-small`) to your resource.
3. **Authentication** — either an API key or a managed identity / Azure CLI credential.

### Configuration

```python
from mem0 import Memory

config = {
    "llm": {
        "provider": "azure_foundry",
        "config": {
            "model": "gpt-4o",                  # deployment name
            "endpoint": "https://<resource>.services.ai.azure.com/models",
            "api_key": "your-api-key",           # or omit for managed identity
            "temperature": 0.1,
            "max_tokens": 2000,
        },
    },
    "embedder": {
        "provider": "azure_foundry",
        "config": {
            "model": "text-embedding-3-small",   # deployment name
            "openai_base_url": "https://<resource>.services.ai.azure.com/models",
            "api_key": "your-api-key",           # or omit for managed identity
            "embedding_dims": 1536,              # must match vector store dims
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 1536,        # must match embedder dims
            "path": "/tmp/mem0_qdrant",
        },
    },
}

memory = Memory.from_config(config)
```

### Environment Variables

Instead of passing credentials in code, set these environment variables:

| Variable | Description |
|---|---|
| `AZURE_FOUNDRY_API_KEY` | API key for the chat completion endpoint |
| `AZURE_FOUNDRY_ENDPOINT` | Chat completion endpoint URL |
| `AZURE_FOUNDRY_EMBEDDING_API_KEY` | API key for the embedding endpoint |
| `AZURE_FOUNDRY_EMBEDDING_ENDPOINT` | Embedding endpoint URL |

When API keys are omitted, the provider automatically uses [`DefaultAzureCredential`](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential) for passwordless auth (managed identity, Azure CLI, etc.).

### Managed Identity (Passwordless) Auth

For deployment on Azure (App Service, Container Apps, AKS, VMs), omit the API key to use managed identity:

```python
config = {
    "llm": {
        "provider": "azure_foundry",
        "config": {
            "model": "gpt-4o",
            "endpoint": "https://<resource>.services.ai.azure.com/models",
            # No api_key → uses DefaultAzureCredential automatically
            "managed_identity_client_id": "...",  # optional, for user-assigned MI
        },
    },
    "embedder": {
        "provider": "azure_foundry",
        "config": {
            "model": "text-embedding-3-small",
            "openai_base_url": "https://<resource>.services.ai.azure.com/models",
            "embedding_dims": 1536,
        },
    },
}
```

You can also set `AZURE_CLIENT_ID` for user-assigned managed identity (natively supported by `azure-identity`).

### Embedding & LLM Model Compatibility

The chat completion model and embedding model are **independent** — any chat model works with any embedding model. The critical compatibility requirement is between the **embedding model** and the **vector store**:

> **`embedding_dims` (embedder config) must equal `embedding_model_dims` (vector store config)**

At startup, mem0 validates this and logs a warning if they don't match:

```
WARNING  Embedding dimension mismatch: embedder config specifies 768 dimensions
         but vector store expects 1536. This will cause errors at runtime.
```

Common embedding dimensions:

| Model | Dimensions |
|---|---|
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
| `text-embedding-ada-002` | 1536 |

### Full Example

```python
import os
from mem0 import Memory

os.environ["AZURE_FOUNDRY_API_KEY"] = "your-key"
os.environ["AZURE_FOUNDRY_ENDPOINT"] = "https://myresource.services.ai.azure.com/models"
os.environ["AZURE_FOUNDRY_EMBEDDING_API_KEY"] = "your-key"
os.environ["AZURE_FOUNDRY_EMBEDDING_ENDPOINT"] = "https://myresource.services.ai.azure.com/models"

memory = Memory.from_config({
    "llm": {
        "provider": "azure_foundry",
        "config": {"model": "gpt-4o"},
    },
    "embedder": {
        "provider": "azure_foundry",
        "config": {
            "model": "text-embedding-3-small",
            "embedding_dims": 1536,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 1536,
            "path": "/tmp/mem0_qdrant",
        },
    },
})

# Add memories
memory.add(
    [{"role": "user", "content": "I prefer dark mode and use VS Code."}],
    user_id="alice",
)

# Search memories
results = memory.search(query="What editor does Alice use?", user_id="alice")
for r in results["results"]:
    print(r["memory"])
```

---

## 🔗 Integrations & Demos

- **ChatGPT with Memory**: Personalized chat powered by Mem0 ([Live Demo](https://mem0.dev/demo))
- **Browser Extension**: Store memories across ChatGPT, Perplexity, and Claude ([Chrome Extension](https://chromewebstore.google.com/detail/onihkkbipkfeijkadecaafbgagkhglop?utm_source=item-share-cb))
- **Langgraph Support**: Build a customer bot with Langgraph + Mem0 ([Guide](https://docs.mem0.ai/integrations/langgraph))
- **CrewAI Integration**: Tailor CrewAI outputs with Mem0 ([Example](https://docs.mem0.ai/integrations/crewai))

## 📚 Documentation & Support

- Full docs: https://docs.mem0.ai
- Community: [Discord](https://mem0.dev/DiG) · [Twitter](https://x.com/mem0ai)
- Contact: founders@mem0.ai

## Citation

We now have a paper you can cite:

```bibtex
@article{mem0,
  title={Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory},
  author={Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal={arXiv preprint arXiv:2504.19413},
  year={2025}
}
```

## ⚖️ License

Apache 2.0 — see the [LICENSE](https://github.com/mem0ai/mem0/blob/main/LICENSE) file for details.