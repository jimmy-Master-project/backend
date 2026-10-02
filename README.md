# Backend — Prompt Population 與 LLM Response MVP

依照 `docs/spec.md` 實作，涵蓋：

```text
建立 Evaluation
呼叫 Prompt Generation Agent 產生 Prompt Population
人工編輯 / 新增 / 刪除 Prompt
設定 Target LLM
執行 EvaluationRun（背景執行緒 + 進度輪詢）
查詢 LLM Response
```

## 安裝

```bash
uv sync
cp .env.example .env
```

需要時編輯 `.env`：

- `PROMPT_AGENT_API_KEY` / `TARGET_LLM_API_KEY`：呼叫雲端 API（OpenAI 等）時填入；本機 vLLM / llama.cpp server 通常留空即可。
- `CORS_ALLOWED_ORIGINS`：預設允許 Vite dev server（`localhost:5173`）。

## 初次啟動

```bash
uv run python manage.py migrate
uv run python manage.py runserver
```

Server 預設在 `http://127.0.0.1:8000`。

## 本地資料與快取

模型、NLTK 資料與可設定的 runtime cache 都會寫入此專案的 `backend/models/` 或 `backend/.cache/`。根目錄的 `start_services.sh` 也會將服務的工具 cache 導向專案內的 `.cache/`，不會使用使用者家目錄作為持久化位置。

## 目錄結構

```text
config/            Django settings / urls
apps/
    evaluations/    Evaluation CRUD
    prompt_agents/  Prompt Generation Agent client + service
    prompts/        PromptPopulation / Prompt models + API
    llms/           Target LLM Config + client
    runs/           EvaluationRun + 執行 service
    responses/      LLMResponse 查詢 API / 外部機器人回傳
    fairness/       LangFair fairness metrics（非同步計算 + 快取）
    personas/       MatrAIx persona 代理 API
    benchmark/      DialogSum counterfactual 驗證
services/           MatrAIx client、camelCase helper
```

## API

所有 JSON 欄位（request body、response body、query string）一律使用 **camelCase**；models 與內部 JSON（`metadata` 等）維持 snake_case，由 serializer / view 明確轉換（未使用全域 camelCase renderer/parser）。
每個 endpoint 同時接受結尾有 `/` 與沒有 `/` 的寫法（`runs/latest/` 例外，只有結尾 `/`）。

例外（key 是「資料」而不是欄位名稱，原樣保留）：

- persona 屬性 dict（`personaAttributes`、`attributes`、`filters` 的 key 是 MatrAIx dimension 名稱，如 `gender_identity`）。
- `PromptSerializer.metadata` / `PromptPopulationSerializer.metadata`：內部儲存的原始 JSON blob（snake_case），原樣輸出。
- fairness-metrics 結果內 `suites.*` 由 LangFair 產生的 metric 名稱 / 原始輸出。
- 列舉值（`text_generation`、`openai_compatible`、`missing_input`…）是值不是 key；DRF 的 `non_field_errors` 錯誤 key 保留。

```text
GET           /api/health/                                         # readiness probe → {"status": "ok"}

# Evaluations
GET/POST      /api/evaluations/                                    # POST 可為 multipart，上傳檔案欄位 `taskFile`
GET/PUT/PATCH/DELETE /api/evaluations/{id}/

# Prompt populations / prompts
POST          /api/prompt-populations/generate/                    # 無 evaluation：body 需 `task`，會先建立 Evaluation
POST          /api/evaluations/{id}/prompt-population/generate/    # 202 → PromptPopulation（同 GET /prompt-populations/{id}/）
POST          /api/evaluations/{id}/prompt-populations/generate/   # 同上（別名）
GET           /api/evaluations/{id}/prompt-population/             # 最新（優先 completed）population 摘要
GET/POST      /api/evaluations/{id}/prompts/                       # 最新 population 的 prompts；POST {text, category}
PATCH/DELETE  /api/evaluations/{id}/prompts/{prompt_id}/           # PATCH {text?, category?}
GET           /api/prompt-populations/{id}/                        # 輪詢生成狀態
GET/POST      /api/prompt-populations/{id}/prompts/                # {count, results}；POST {text, category}
GET/PUT/PATCH/DELETE /api/prompts/{id}/

# Target LLM
GET/PUT       /api/evaluations/{id}/llm-config/
POST          /api/evaluations/{id}/llm-config/test-connection/

# Runs
POST          /api/evaluations/{id}/runs/                          # {promptPopulationId?, external?, resumeRunId?} → 201 RunStatus
GET           /api/evaluations/{id}/runs/latest/                   # RunStatus（無 run 時 404）
GET           /api/runs/{run_id}/status/                           # RunStatus

# Responses
GET           /api/evaluations/{id}/responses/?keyword=&category=&generationIndex=&page=&pageSize=
GET           /api/runs/{run_id}/responses/?promptId=&category=    # {count, results}
POST          /api/runs/{run_id}/responses/                        # 接收外部機器人回答，可選擇立即計算 LangFair
DELETE        /api/runs/{run_id}/responses/{response_id}/

# Fairness (LangFair, 非同步)
POST          /api/runs/{run_id}/fairness-metrics/                 # {suites?, options?} → 200（已快取）或 202（計算中）
GET           /api/runs/{run_id}/fairness-metrics/?suite=a&suite=b&options=<json>   # 不計算，只回傳狀態/快取

# Personas（代理 MatrAIx，backend↔MatrAIx 維持 MatrAIx 自己的格式）
GET           /api/persona/attributes/?pool=                       # {pool, attributes: [{key, label, category, description?, values?}]}
POST          /api/persona/sample/                                 # {sampleSize, selectedAttributes, pool?, filters?, stratify?, seed?}

# Benchmark
POST          /api/benchmark/dialogsum/                            # {domain, pairCount, seed} → 202 {evaluationId, runId, pairCount, promptCount, status}
```

主要 response 形狀：

```text
RunStatus            {id, status, totalPrompts, completedPrompts, totalGenerations, completedGenerations,
                      failedGenerations, progress, errorMessage, external, heartbeatAt}
PromptPopulation     {id, evaluationId, name, provider, modelName, temperature, requestedPromptCount,
                      actualPromptCount, additionalInstruction, metadata, status, errorMessage, createdAt}
Population summary   {populationId, status, errorMessage, generationModel, provider, temperature, requested,
                      generated, personaBased, persona, personaAttributes, stratifiedBy, seed, fairness,
                      generatingPopulationId}
Prompt (evaluation)  {id, index, text, category, personaId, personaAttributes, missingAttributes, fairness}
```

Population summary 的 `generatingPopulationId` 是最新一個仍在生成中（pending/generating）的 population id（可能比 `populationId` 新），前端重新整理頁面後可據此繼續輪詢 `GET /api/prompt-populations/{id}/`。伺服器重啟後沒有 worker thread 的生成中 population，在讀取摘要或狀態時會被標成 `failed`。

產生 Prompt Population 的 body：

```json
{
  "provider": "openai_compatible",
  "modelName": "",
  "baseUrl": "",
  "promptCount": 20,
  "temperature": 0.8,
  "additionalInstruction": "",
  "persona": {
    "enabled": true,
    "pool": "",
    "selectedAttributes": ["gender_identity"],
    "filters": {},
    "stratify": null,
    "seed": null,
    "counterfactualEnabled": true
  }
}
```

### 外部機器人

先以 `POST /api/evaluations/{id}/runs/`、body `{"external": true}` 建立 run（回傳 RunStatus，`id` 即 run id），再把回答送到 response endpoint。

若設定 `EXTERNAL_RESPONSE_API_KEY`，request 必須包含 `X-API-Key` header：

```http
X-API-Key: your-shared-key
```

```json
{
  "responses": [
    {
      "promptId": 123,
      "response": "robot answer",
      "generationIndex": 1,
      "inputTokens": 10,
      "outputTokens": 20,
      "latencyMs": 350
    }
  ],
  "calculate": true,
  "suites": ["toxicity", "stereotype", "counterfactual"],
  "options": {}
}
```

回應：`{stored, created, updated, responseIds, metrics?}`。API 會先將回答寫入 `LLMResponse`；`calculate: true` 時同步使用 `LangFairMetricService` 計算並放在 `metrics`。若不帶 `calculate` 或設為 `false`，可在資料全部送完後呼叫 `POST /api/runs/{run_id}/fairness-metrics/` 啟動計算，再以 GET 輪詢結果。

## 備註

- `provider` 支援 `openai`（`https://api.openai.com/v1`）、`evaair`（長榮聊天機器人，經由專案根目錄的 `evaair-bot` 服務，預設 `http://127.0.0.1:8768/v1`）、`virgin`（Virgin Atlantic Concierge AI，經由專案根目錄的 `browser-bots` 服務以瀏覽器操作，預設 `http://127.0.0.1:8769/virgin/v1`）、`priceline`（Priceline Penny，經由專案根目錄的 `priceline-bot` 服務，預設 `http://127.0.0.1:8770/v1`）與 `openai_compatible`（vLLM、llama.cpp 等 OpenAI 相容服務），其他值會回 400。全部都走 OpenAI SDK；聊天機器人 provider 不會收到 OpenAI key。
- Prompt Generation 與 Target LLM 呼叫都使用 OpenAI SDK 的 `base_url` 參數指向自架 endpoint。
- EvaluationRun 用背景 thread 執行（非 Celery），符合 MVP 範圍；前端可用 `GET /api/runs/{id}/status/` 輪詢進度。
