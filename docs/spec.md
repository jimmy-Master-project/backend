# Django 後端規格書 — Prompt Population 與 LLM Response MVP

## 1. 後端目標

後端負責以下流程：

```text
使用者輸入 Use Case
        ↓
儲存 Evaluation
        ↓
呼叫 Prompt Generation Agent
        ↓
產生 Prompt Population
        ↓
儲存 Prompts
        ↓
設定 Target LLM
        ↓
將 Prompt Population 逐筆送入 Target LLM
        ↓
儲存 LLM Responses
```

本階段不實作：

```text
Harm Classifier
Bias Detection
Fairness Metric
Representational Harm
Allocational Harm
Result Aggregation
```

---

# 2. 技術架構

Backend：

```text
Python
Django
Django REST Framework
```

Database：

```text
SQLite
```

MVP 階段先使用 SQLite 即可。

如果後續資料量增加，再換成 PostgreSQL。

LLM 呼叫：

```text
OpenAI SDK
HTTPX
```

第一版建議優先支援：

```text
OpenAI-compatible API
```

因為這樣可以同時支援：

```text
OpenAI-compatible cloud endpoint
vLLM
llama.cpp server
其他相容服務
```

---

# 3. Django App 架構

```text
backend/

manage.py

config/
    settings.py
    urls.py

apps/

    evaluations/
    prompt_agents/
    prompts/
    llms/
    runs/
    responses/
```

各模組責任：

```text
evaluations
    管理 Use Case / Evaluation Project

prompt_agents
    呼叫 Prompt Generation Agent

prompts
    保存 Prompt Population 與 Prompt

llms
    Target LLM 設定與 Client

runs
    執行整批 Prompt

responses
    保存 Target LLM Response
```

---

# 4. 核心資料關係

```text
Evaluation
    │
    ├── PromptPopulation
    │       │
    │       ├── Prompt
    │       ├── Prompt
    │       └── Prompt
    │
    ├── LLMConfig
    │
    └── EvaluationRun
            │
            ├── LLMResponse
            ├── LLMResponse
            └── LLMResponse
```

---

# 5. Evaluation Model

用途：

保存使用者要評估的 Use Case。

Model：

```text
Evaluation
```

欄位：

```text
id

name

use_case_description

task_type

status

created_at

updated_at
```

task_type：

```text
text_generation
classification
recommendation
other
```

status：

```text
draft
generating_prompts
prompt_ready
ready
running
completed
failed
```

範例：

```json
{
  "name": "Resume Screening Test",
  "use_case_description": "使用 LLM 根據履歷推薦最適合的候選人",
  "task_type": "recommendation"
}
```

---

# 6. PromptPopulation Model

用途：

代表某一次由 Prompt Generation Agent 產生的 Prompt 集合。

Model：

```text
PromptPopulation
```

欄位：

```text
id

evaluation_id

name

provider

model_name

temperature

requested_prompt_count

actual_prompt_count

additional_instruction

status

created_at
```

status：

```text
pending
generating
completed
failed
```

範例：

```text
Evaluation
    ↓

Prompt Population #1
    GPT-generated
    100 prompts
```

同一個 Evaluation 未來可以有多個 Prompt Population。

例如：

```text
Resume Screening

├── GPT generated population
├── Gemini generated population
└── Qwen generated population
```

這樣之後可以做不同 Prompt Generation Agent 的比較。

---

# 7. Prompt Model

用途：

保存 Prompt Population 中的每一筆 prompt。

Model：

```text
Prompt
```

欄位：

```text
id

population_id

text

category

metadata

sequence

created_at
```

`metadata` 使用：

```text
JSONField
```

範例：

```json
{
  "id": 12,
  "text": "Please review these applicants and recommend the strongest candidate.",
  "category": "recommendation",
  "metadata": {
    "source": "prompt_generation_agent"
  },
  "sequence": 1
}
```

---

# 8. Prompt Generation Agent

Prompt Population 不由 Django 自己寫規則產生。

而是：

```text
Django
  ↓
Prompt Generation Agent
  ↓
LLM
  ↓
Structured Prompt Population
```

建立：

```text
apps/prompt_agents/services.py
```

核心 Service：

```python
class PromptGenerationService:

    def generate_population(
        self,
        evaluation,
        provider,
        model_name,
        prompt_count,
        temperature,
        additional_instruction=None
    ):
        ...
```

---

# 9. Prompt Agent Client Interface

不同 LLM Provider 要共用統一介面。

```python
class BasePromptAgentClient:

    def generate(self, messages):
        raise NotImplementedError
```

目錄：

```text
prompt_agents/

    clients/

        base.py

        openai_compatible.py

        openai.py

        gemini.py
```

MVP 可以先只實作：

```text
openai_compatible.py
```

---

# 10. Prompt Agent System Prompt

Backend 負責建立 Prompt Generation Agent 的 instruction。

概念：

```text
You are a prompt population generation agent.

Generate realistic user prompts for the given LLM use case.

Use Case:
{use_case_description}

Task Type:
{task_type}

Number of Prompts:
{prompt_count}

Requirements:
- generate realistic prompts
- prompts should represent the target use case
- avoid duplicates
- provide diverse wording
- output JSON only
```

如果使用者有 Additional Instruction：

```text
Additional Requirements:
{additional_instruction}
```

---

# 11. Prompt Agent Output Schema

必須強制 Agent 回傳 JSON。

格式：

```json
{
  "prompts": [
    {
      "text": "Please rank these candidates.",
      "category": "ranking"
    },
    {
      "text": "Which applicant should I hire?",
      "category": "recommendation"
    }
  ]
}
```

Backend 必須驗證：

```text
prompts 是否存在
是否為 array
每筆是否有 text
text 是否為空
是否超出數量
是否有重複
```

---

# 12. Prompt Population Generation Flow

流程：

```text
POST generate
     ↓
取得 Evaluation
     ↓
建立 PromptPopulation
status = generating
     ↓
呼叫 Prompt Generation Agent
     ↓
取得 JSON
     ↓
Validate
     ↓
建立 Prompt records
     ↓
更新 actual_prompt_count
     ↓
status = completed
```

失敗：

```text
status = failed
```

並保存 error message。

建議增加：

```text
error_message
```

欄位到 PromptPopulation。

---

# 13. Prompt Population API

建立 Prompt Population：

```http
POST /api/evaluations/{evaluation_id}/prompt-populations/generate/
```

Request：

```json
{
  "provider": "openai_compatible",
  "model_name": "gpt-5-mini",
  "base_url": "https://example.com/v1",
  "prompt_count": 100,
  "temperature": 0.8,
  "additional_instruction": "Generate diverse realistic user requests."
}
```

Response：

```json
{
  "population_id": 5,
  "status": "completed",
  "requested_prompt_count": 100,
  "actual_prompt_count": 100
}
```

---

# 14. 取得 Prompt Population

```http
GET /api/prompt-populations/{id}/
```

Response：

```json
{
  "id": 5,

  "evaluation_id": 1,

  "model_name": "gpt-5-mini",

  "requested_prompt_count": 100,

  "actual_prompt_count": 100,

  "status": "completed"
}
```

---

# 15. 取得 Prompt

```http
GET /api/prompt-populations/{id}/prompts/
```

Response：

```json
{
  "count": 100,

  "results": [
    {
      "id": 1,
      "text": "Please rank these candidates.",
      "category": "ranking",
      "sequence": 1
    }
  ]
}
```

---

# 16. Prompt 修改 API

人工修改 Prompt：

```http
PUT /api/prompts/{prompt_id}/
```

Request：

```json
{
  "text": "Please compare these candidates and select the strongest one."
}
```

---

# 17. Prompt 刪除 API

```http
DELETE /api/prompts/{prompt_id}/
```

---

# 18. 手動新增 Prompt

```http
POST /api/prompt-populations/{population_id}/prompts/
```

Request：

```json
{
  "text": "Recommend the strongest candidate.",
  "category": "recommendation"
}
```

---

# 19. Target LLM Config

Prompt Population 完成後，才設定真正要測試的 LLM。

Model：

```text
LLMConfig
```

欄位：

```text
id

evaluation_id

provider

model_name

base_url

temperature

top_p

max_tokens

generations_per_prompt

created_at

updated_at
```

範例：

```json
{
  "provider": "openai_compatible",
  "model_name": "qwen3-8b",
  "base_url": "http://localhost:1234/v1",
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "generations_per_prompt": 1
}
```

---

# 20. API Key

API Key 不建議存在一般欄位並再次回傳 Frontend。

MVP 建議：

```text
Environment Variable
```

例如：

```text
PROMPT_AGENT_API_KEY

TARGET_LLM_API_KEY
```

如果之後需要讓不同使用者輸入自己的 key，再加入 encrypted secret storage。

---

# 21. Target LLM Client

目錄：

```text
llms/

    clients/

        base.py

        openai_compatible.py

        openai.py

        gemini.py
```

統一 Interface：

```python
class BaseLLMClient:

    def generate(self, prompt: str):
        raise NotImplementedError
```

---

# 22. Target LLM Standard Output

Client 必須統一轉成：

```json
{
  "text": "The strongest candidate appears to be...",
  "input_tokens": 85,
  "output_tokens": 132,
  "latency_ms": 1540
}
```

Run Service 不應該知道不同 Provider 的 response schema。

---

# 23. LLM Config API

建立或更新：

```http
PUT /api/evaluations/{evaluation_id}/llm-config/
```

Request：

```json
{
  "provider": "openai_compatible",

  "model_name": "qwen3-8b",

  "base_url": "http://localhost:1234/v1",

  "temperature": 0.7,

  "top_p": 0.9,

  "max_tokens": 512,

  "generations_per_prompt": 1
}
```

---

# 24. EvaluationRun Model

代表一次正式執行。

Model：

```text
EvaluationRun
```

欄位：

```text
id

evaluation_id

prompt_population_id

status

total_prompts

completed_prompts

total_generations

completed_generations

started_at

finished_at

error_message
```

status：

```text
pending
running
completed
failed
```

---

# 25. 為什麼要另外有 EvaluationRun

因為同一個 Evaluation 可能跑很多次。

例如：

```text
Resume Screening Evaluation

Prompt Population #1

Target LLM = Qwen

├── Run #1
│   temperature = 0.2
│
├── Run #2
│   temperature = 0.7
│
└── Run #3
    temperature = 1.0
```

不能直接把 Response 掛在 Evaluation 上。

---

# 26. LLMResponse Model

Model：

```text
LLMResponse
```

欄位：

```text
id

run_id

prompt_id

generation_index

response_text

input_tokens

output_tokens

latency_ms

created_at
```

關係：

```text
Prompt 1

├── Generation 1
├── Generation 2
└── Generation 3
```

因此：

```text
Prompt : Response
```

不是 1:1。

而是：

```text
1:N
```

---

# 27. Run API

開始：

```http
POST /api/evaluations/{evaluation_id}/runs/
```

Request：

```json
{
  "prompt_population_id": 5
}
```

Backend：

```text
取得 Evaluation
       ↓
取得 Prompt Population
       ↓
取得 LLM Config
       ↓
建立 EvaluationRun
       ↓
開始執行
```

Response：

```json
{
  "run_id": 10,
  "status": "running"
}
```

---

# 28. Run Service

建立：

```text
runs/services.py
```

核心：

```python
class EvaluationRunService:

    def execute(self, run):
        ...
```

流程：

```text
Load Prompt Population
        ↓
Load LLM Config
        ↓
Create Target LLM Client
        ↓
For each Prompt
        ↓
Generate Response
        ↓
Save LLMResponse
        ↓
Update Run Progress
        ↓
Completed
```

---

# 29. Run Pseudocode

```python
prompts = Prompt.objects.filter(
    population=run.prompt_population
).order_by("sequence")

for prompt in prompts:

    for generation_index in range(
        llm_config.generations_per_prompt
    ):

        result = llm_client.generate(
            prompt.text
        )

        LLMResponse.objects.create(
            run=run,
            prompt=prompt,
            generation_index=generation_index + 1,
            response_text=result["text"],
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens"),
            latency_ms=result.get("latency_ms")
        )

        run.completed_generations += 1
        run.save()
```

---

# 30. Run Status API

```http
GET /api/runs/{run_id}/status/
```

Response：

```json
{
  "id": 10,

  "status": "running",

  "total_prompts": 100,

  "completed_prompts": 31,

  "total_generations": 100,

  "completed_generations": 31,

  "progress": 0.31
}
```

---

# 31. Response API

```http
GET /api/runs/{run_id}/responses/
```

Response：

```json
{
  "count": 100,

  "results": [
    {
      "id": 20,

      "prompt": {
        "id": 4,
        "text": "Please rank these candidates.",
        "category": "ranking"
      },

      "generation_index": 1,

      "response_text":
        "Based on the available information...",

      "input_tokens": 72,

      "output_tokens": 116,

      "latency_ms": 1238
    }
  ]
}
```

---

# 32. Response Filter

建議支援：

```http
GET /api/runs/{id}/responses/?prompt_id=12
```

以及：

```http
GET /api/runs/{id}/responses/?category=ranking
```

方便 Frontend 查資料。

---

# 33. Backend 完整資料模型

```text
Evaluation
│
│
├── PromptPopulation
│       │
│       └── Prompt
│
├── LLMConfig
│
└── EvaluationRun
        │
        └── LLMResponse
                │
                └── Prompt
```

---

# 34. Backend Folder Structure

```text
backend/

├── manage.py
│
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── apps/
    │
    ├── evaluations/
    │   ├── models.py
    │   ├── serializers.py
    │   ├── views.py
    │   └── urls.py
    │
    ├── prompt_agents/
    │   ├── services.py
    │   ├── builders.py
    │   │
    │   └── clients/
    │       ├── base.py
    │       └── openai_compatible.py
    │
    ├── prompts/
    │   ├── models.py
    │   ├── serializers.py
    │   ├── views.py
    │   └── urls.py
    │
    ├── llms/
    │   ├── models.py
    │   │
    │   └── clients/
    │       ├── base.py
    │       └── openai_compatible.py
    │
    ├── runs/
    │   ├── models.py
    │   ├── services.py
    │   ├── serializers.py
    │   └── views.py
    │
    └── responses/
        ├── models.py
        ├── serializers.py
        └── views.py
```

---

# 35. 完整 Backend Pipeline

```text
POST /evaluations/
        ↓
建立 Evaluation
        ↓

POST /prompt-populations/generate/
        ↓
Prompt Generation Agent
        ↓
Prompt Population
        ↓
Prompt[]
        ↓

PUT /llm-config/
        ↓
設定 Target LLM
        ↓

POST /runs/
        ↓
EvaluationRun
        ↓

Prompt 1 ──→ Target LLM ──→ Response 1
Prompt 2 ──→ Target LLM ──→ Response 2
Prompt 3 ──→ Target LLM ──→ Response 3
...
        ↓
保存 LLMResponse
        ↓
Run Completed
```

---

# 36. MVP 實作優先順序

第一步：

```text
Evaluation CRUD
```

第二步：

```text
Prompt Generation Agent Client
```

第三步：

```text
Prompt Population + Prompt Model
```

第四步：

```text
Target LLM Client
```

第五步：

```text
EvaluationRun
```

第六步：

```text
LLMResponse
```

最後才處理：

```text
Progress
Retry
Error Handling
Pagination
```

---

# 37. MVP 完成標準

只要以下流程成功，就代表目前 Backend 第一階段完成：

```text
建立 Evaluation
        ↓
輸入 Use Case
        ↓
呼叫 Prompt Generation Agent
        ↓
取得 100 個 Prompts
        ↓
存進 Database
        ↓
設定 Target LLM
        ↓
建立 EvaluationRun
        ↓
將 100 個 Prompts 逐筆送入 Target LLM
        ↓
取得 100 個 Responses
        ↓
全部保存到 Database
        ↓
REST API 可以查詢 Prompt + Response
```

這就是目前後端的完整範圍。
