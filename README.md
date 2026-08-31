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
```

## 備註

- `provider` 目前只實作 `openai_compatible`（同時涵蓋 OpenAI cloud endpoint、vLLM、llama.cpp 等相容服務），對應 spec 第一版建議。
- Prompt Generation 與 Target LLM 呼叫都使用 OpenAI SDK 的 `base_url` 參數指向自架 endpoint。
- EvaluationRun 用背景 thread 執行（非 Celery），符合 MVP 範圍；前端可用 `GET /api/runs/{id}/status/` 輪詢進度。
- 尚未實作：Harm Classifier、Bias Detection、Fairness Metric 等（依 spec 本階段排除）。
