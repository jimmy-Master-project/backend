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

模型、NLTK 資料與可設定的 runtime cache 都會寫入此專案的 `backend/models/` 或 `backend/.cache/`。根目錄的 `start_services.sh` 也會將三個服務的工具 cache 導向專案內的 `.cache/`，不會使用使用者家目錄作為持久化位置。

## 目錄結構

```text
config/            Django settings / urls
apps/
    evaluations/    Evaluation CRUD
    prompt_agents/  Prompt Generation Agent client + service
    prompts/        PromptPopulation / Prompt models + API
    llms/           Target LLM Config + client
    runs/           EvaluationRun + 執行 service
    responses/      LLMResponse 查詢 API
```

## API

```text
GET/POST      /api/evaluations/
GET/PUT/DELETE /api/evaluations/{id}/

POST          /api/evaluations/{evaluation_id}/prompt-populations/generate/
GET           /api/prompt-populations/{id}/
GET/POST      /api/prompt-populations/{id}/prompts/
PUT/DELETE    /api/prompts/{id}/

GET/PUT       /api/evaluations/{evaluation_id}/llm-config/

POST          /api/evaluations/{evaluation_id}/runs/
GET           /api/runs/{run_id}/status/
GET           /api/runs/{run_id}/responses/?prompt_id=&category=
POST          /api/runs/{run_id}/responses/       # 接收外部機器人回答，可選擇立即計算 LangFair
GET/POST       /api/runs/{run_id}/fairness-metrics/
```

外部機器人回傳結果後，可呼叫 response endpoint：

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

API 會先將回答寫入 `LLMResponse`，再使用相同的 `LangFairMetricService` 計算。若不帶 `calculate` 或設為 `false`，可在資料全部送完後呼叫 `GET /api/runs/{run_id}/fairness-metrics/` 計算一次。

## 備註

- `provider` 目前只實作 `openai_compatible`（同時涵蓋 OpenAI cloud endpoint、vLLM、llama.cpp 等相容服務），對應 spec 第一版建議。
- `provider=custom` 時，`baseUrl` 會被當成外部機器人的 POST URL。請求格式為 `{"prompt":"...","model":"...","temperature":0.7,"top_p":1.0,"max_tokens":512}`；回應可為純文字，或 JSON 的 `text`、`response`、`output`、`message.content` 欄位。`apiKey` 會以 Bearer token 放入請求，且不會由 GET API 回傳。
- Prompt Generation 與 Target LLM 呼叫都使用 OpenAI SDK 的 `base_url` 參數指向自架 endpoint。
- EvaluationRun 用背景 thread 執行（非 Celery），符合 MVP 範圍；前端可用 `GET /api/runs/{id}/status/` 輪詢進度。
- 尚未實作：Harm Classifier、Bias Detection、Fairness Metric 等（依 spec 本階段排除）。
