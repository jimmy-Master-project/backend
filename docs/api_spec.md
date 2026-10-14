# LLM Fairness Pipeline API Specification

## 1. 共通規則

- Base URL：`http://localhost:8000/api`
- Request format：`application/json`
- Response format：`application/json`
- URL 最後的 `/` 可省略。
- 目前 API 沒有啟用登入驗證；部署到公開環境時應放在 API Gateway 或設定驗證機制。

### 一般錯誤格式

```json
{
  "field": ["錯誤原因"]
}
```

或：

```json
{
  "error": "錯誤原因"
}
```

## 2. Evaluation API

### 建立 Evaluation

```http
POST /api/evaluations/
```

#### Request

```json
{
  "name": "履歷推薦公平性評估",
  "useCaseDescription": "使用 LLM 分析履歷並推薦候選人",
  "taskType": "recommendation",
  "language": "en",
  "recommendationItems": [
    {
      "name": "Candidate A",
      "description": "Backend engineer"
    },
    {
      "name": "Candidate B",
      "description": "Frontend engineer"
    }
  ],
  "taskObjectives": []
}
```

#### 欄位

| 欄位 | 必填 | 說明 |
|---|---:|---|
| `name` | 是 | Evaluation 名稱 |
| `useCaseDescription` | 否 | 任務描述；沒有 `taskObjectives` 時必填 |
| `taskType` | 是 | `text_generation`、`classification`、`recommendation`、`other` |
| `language` | 否 | Evaluation 語言 |
| `classificationLabels` | 否 | 分類任務的兩個分類標籤 |
| `recommendationItems` | 否 | recommendation 任務的候選項目，至少兩個 |
| `taskObjectives` | 否 | 任務目標陣列 |

#### Response `201`

```json
{
  "id": 1,
  "name": "履歷推薦公平性評估",
  "useCaseDescription": "使用 LLM 分析履歷並推薦候選人",
  "taskType": "recommendation",
  "language": "zh",
  "classificationLabels": [],
  "recommendationItems": [],
  "taskObjectives": [],
  "status": "draft",
  "promptCount": 0,
  "createdAt": "2026-09-15T10:00:00Z",
  "updatedAt": "2026-09-15T10:00:00Z"
}
```

### 上傳 Task File 建立 Evaluation

```http
POST /api/evaluations/
Content-Type: multipart/form-data
```

Form fields：

| 欄位 | 必填 | 說明 |
|---|---:|---|
| `name` | 是 | Evaluation 名稱 |
| `taskType` | 是 | 任務類型 |
| `task_file` | 是 | 任務檔案 |

### Evaluation 列表

```http
GET /api/evaluations/
```

Response：Evaluation object array。

### 讀取 Evaluation

```http
GET /api/evaluations/{evaluation_id}/
```

Response：單一 Evaluation object。

### 更新 Evaluation

```http
PUT /api/evaluations/{evaluation_id}/
PATCH /api/evaluations/{evaluation_id}/
```

Request 格式與建立 Evaluation 相同，可只傳要修改的欄位。

### 刪除 Evaluation

```http
DELETE /api/evaluations/{evaluation_id}/
```

Response：`204 No Content`

## 3. Prompt Population API

### 產生 Prompt Population

```http
POST /api/evaluations/{evaluation_id}/prompt-populations/generate/
```

#### Request

```json
{
  "provider": "openai_compatible",
  "model": "gpt-4o-mini",
  "base_url": "http://localhost:8001/v1",
  "promptCount": 100,
  "temperature": 0.8,
  "additionalInstructions": "請產生多元化案例",
  "persona": {
    "enabled": true,
    "selected_attributes": ["gender", "age"],
    "counterfactual_enabled": true,
    "stratify": "gender",
    "sampling_seed": 42
  }
}
```

`model`、`promptCount`、`additionalInstructions` 也可使用 `model_name`、`prompt_count`、`additional_instruction`。

#### Response `202`

```json
{
  "population_id": 10,
  "status": "generating",
  "requested_prompt_count": 100
}
```

也支援不帶 `evaluation_id` 的快速建立：

```http
POST /api/prompt-populations/generate/
```

此時 request 必須包含 `task`。

### 讀取 Prompt Population

```http
GET /api/prompt-populations/{population_id}/
```

#### Response

```json
{
  "id": 10,
  "evaluation_id": 1,
  "name": "",
  "provider": "openai_compatible",
  "model_name": "gpt-4o-mini",
  "temperature": 0.8,
  "requested_prompt_count": 100,
  "actual_prompt_count": 100,
  "additional_instruction": "",
  "metadata": {},
  "status": "completed",
  "error_message": "",
  "created_at": "2026-09-15T10:00:00Z"
}
```

### 讀取 Evaluation 的 Prompt Population 摘要

```http
GET /api/evaluations/{evaluation_id}/prompt-population/
```

#### Response

```json
{
  "generationModel": "gpt-4o-mini",
  "temperature": 0.8,
  "requested": 100,
  "generated": 100,
  "personaBased": true,
  "persona": {},
  "personaAttributes": ["gender", "age"],
  "stratifiedBy": "gender",
  "seed": 42,
  "fairness": {}
}
```

### 讀取 Evaluation Prompts

```http
GET /api/evaluations/{evaluation_id}/prompts/
```

#### Response

```json
[
  {
    "id": "123",
    "index": 1,
    "prompt": "請推薦適合的候選人",
    "text": "請推薦適合的候選人",
    "category": "gender",
    "persona_id": "p001",
    "persona_attributes": {},
    "missing_attributes": [],
    "fairness": {}
  }
]
```

### 新增 Prompt

```http
POST /api/evaluations/{evaluation_id}/prompts/
```

```json
{
  "prompt": "請推薦適合的候選人",
  "category": "gender"
}
```

Response：Prompt object，status `201`。

也可使用：

```http
POST /api/prompt-populations/{population_id}/prompts/
```

此 endpoint 使用：

```json
{
  "text": "請推薦適合的候選人",
  "category": "gender"
}
```

### 修改 Prompt

```http
PATCH /api/evaluations/{evaluation_id}/prompts/{prompt_id}/
PATCH /api/prompts/{prompt_id}/
```

```json
{
  "text": "修改後的 prompt",
  "category": "profession"
}
```

### 刪除 Prompt

```http
DELETE /api/evaluations/{evaluation_id}/prompts/{prompt_id}/
DELETE /api/prompts/{prompt_id}/
```

Response：`204 No Content`

### 讀取 Population Prompts

```http
GET /api/prompt-populations/{population_id}/prompts/
```

Response：

```json
{
  "count": 1,
  "results": [
    {
      "id": 123,
      "text": "請推薦適合的候選人",
      "category": "gender",
      "metadata": {},
      "sequence": 1,
      "created_at": "2026-09-15T10:00:00Z"
    }
  ]
}
```

## 4. Target LLM Config API

### 讀取設定

```http
GET /api/evaluations/{evaluation_id}/llm-config/
```

#### Response

```json
{
  "id": 1,
  "provider": "custom",
  "modelName": "robot-v1",
  "baseUrl": "https://robot.example.com/generate",
  "temperature": 0.7,
  "topP": 1.0,
  "maxTokens": 512,
  "generationsPerPrompt": 1,
  "createdAt": "2026-09-15T10:00:00Z",
  "updatedAt": "2026-09-15T10:00:00Z"
}
```

`apiKey` 不會出現在 response。

### 儲存設定

```http
PUT /api/evaluations/{evaluation_id}/llm-config/
```

```json
{
  "provider": "custom",
  "modelName": "robot-v1",
  "baseUrl": "https://robot.example.com/generate",
  "apiKey": "robot-secret",
  "temperature": 0.7,
  "topP": 1.0,
  "maxTokens": 512,
  "generationsPerPrompt": 1
}
```

支援的 `provider`：

- `openai_compatible`
- `openai`
- `custom`

### 測試 Target LLM 連線

```http
POST /api/evaluations/{evaluation_id}/llm-config/test-connection/
```

#### Response 成功

```json
{
  "success": true,
  "message": "Connection successful."
}
```

#### Response 失敗

```json
{
  "success": false,
  "message": "錯誤原因"
}
```

## 5. Custom Target LLM 外部 API

當 `provider` 設為 `custom` 時，系統會對 `baseUrl` 發送：

```http
POST https://robot.example.com/generate
Content-Type: application/json
Authorization: Bearer robot-secret
```

### Request

```json
{
  "prompt": "請回答以下問題",
  "model": "robot-v1",
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 512
}
```

### Response

外部機器人可回傳純文字：

```text
這是機器人的回答
```

或 JSON：

```json
{
  "text": "這是機器人的回答",
  "input_tokens": 10,
  "output_tokens": 20
}
```

也支援 `response`、`output`、`message.content`：

```json
{
  "message": {
    "content": "這是機器人的回答"
  }
}
```

外部 API 回傳非 `2xx`，或沒有可讀取的文字時，該次 generation 會失敗。

## 6. Evaluation Run API

### 啟動 Run

```http
POST /api/evaluations/{evaluation_id}/runs/
```

#### Request

```json
{
  "prompt_population_id": 10
}
```

`prompt_population_id` 可省略，系統會自動選擇最新完成的 Prompt Population。

#### Response `201`

```json
{
  "run_id": 20,
  "status": "running"
}
```

### 查詢 Run 狀態

```http
GET /api/runs/{run_id}/status/
```

#### Response

```json
{
  "id": 20,
  "status": "completed",
  "total_prompts": 100,
  "completed_prompts": 100,
  "total_generations": 100,
  "completed_generations": 100,
  "progress": 1.0,
  "error_message": ""
}
```

## 7. 外部 Target LLM Response API

外部機器人完成回答後，將回答寫回指定 Run：

```http
POST /api/runs/{run_id}/responses/
Content-Type: application/json
```

若設定 `EXTERNAL_RESPONSE_API_KEY`，需加入：

```http
X-API-Key: your-shared-key
```

### Request

```json
{
  "responses": [
    {
      "promptId": 123,
      "response": "這是外部機器人的回答",
      "generationIndex": 1,
      "inputTokens": 10,
      "outputTokens": 20,
      "latencyMs": 350
    }
  ],
  "calculate": true,
  "suites": [
    "toxicity",
    "stereotype",
    "counterfactual"
  ],
  "options": {}
}
```

### Response `201`

```json
{
  "stored": 1,
  "responseIds": [456],
  "metrics": {
    "runId": 20,
    "responseCount": 1,
    "availableSuites": [
      "toxicity",
      "stereotype",
      "classification",
      "counterfactual",
      "recommendation"
    ],
    "selectedSuites": ["toxicity"],
    "suites": {}
  }
}
```

當 `calculate` 為 `false` 或省略時，只儲存 response，不立即執行 LangFair。

### 分批回傳

```json
{
  "responses": [
    {
      "promptId": 123,
      "response": "robot answer",
      "generationIndex": 1
    }
  ],
  "calculate": false
}
```

所有 response 送完後，再呼叫 LangFair API。

## 8. Response 查詢 API

### 查詢 Evaluation 最新 Run 的 Responses

```http
GET /api/evaluations/{evaluation_id}/responses/
```

Query parameters：

| 參數 | 說明 |
|---|---|
| `keyword` | 搜尋 prompt 或 response 文字 |
| `category` | 依 prompt category 篩選 |
| `generationIndex` | 依 generation index 篩選 |
| `page` | 頁碼，預設 `1` |
| `pageSize` | 每頁數量，預設 `20` |

#### Response

```json
{
  "items": [
    {
      "id": "456",
      "promptId": "123",
      "promptIndex": 1,
      "promptText": "請回答以下問題",
      "promptCategory": "general",
      "response": "機器人的回答",
      "generationIndex": 1,
      "model": "robot-v1",
      "inputTokens": 10,
      "outputTokens": 20,
      "latencyMs": 350
    }
  ],
  "total": 1,
  "model": "robot-v1",
  "promptCount": 100,
  "runId": 20,
  "taskType": "text_generation",
  "language": "en",
  "classificationLabels": []
}
```

### 查詢 Run 的 Responses

```http
GET /api/runs/{run_id}/responses/
```

Query parameters：`prompt_id`、`category`。

#### Response

```json
{
  "count": 1,
  "results": [
    {
      "id": 456,
      "prompt": {
        "id": 123,
        "text": "請回答以下問題",
        "category": "general"
      },
      "generation_index": 1,
      "response_text": "機器人的回答",
      "input_tokens": 10,
      "output_tokens": 20,
      "latency_ms": 350,
      "created_at": "2026-09-15T10:00:00Z"
    }
  ]
}
```

## 9. LangFair Metrics API

### 執行 LangFair

```http
GET /api/runs/{run_id}/fairness-metrics/
```

可用 query parameter：

```http
GET /api/runs/20/fairness-metrics/?suite=toxicity&suite=stereotype
```

也可使用 POST 傳入設定：

```http
POST /api/runs/{run_id}/fairness-metrics/
```

#### Request

```json
{
  "suites": [
    "toxicity",
    "stereotype",
    "counterfactual"
  ],
  "options": {
    "toxicity": {
      "classifiers": ["detoxify_unbiased"],
      "byPrompt": true,
      "returnData": false
    },
    "stereotype": {
      "byPrompt": true,
      "returnData": false
    }
  }
}
```

### 支援 Suites

```text
toxicity
stereotype
classification
counterfactual
recommendation
```

### Response

```json
{
  "runId": 20,
  "responseCount": 100,
  "availableSuites": [
    "toxicity",
    "stereotype",
    "classification",
    "counterfactual",
    "recommendation"
  ],
  "selectedSuites": ["toxicity"],
  "taskType": "text_generation",
  "suites": {
    "toxicity": {
      "Toxic Fraction": 0.12,
      "Expected Maximum Toxicity": 0.35
    }
  }
}
```

LangFair 會以 Run 目前已儲存的所有 `LLMResponse` 作為計算輸入，並使用 cache 避免相同資料重複計算。

`options.toxicity.classifiers` 只接受 `detoxify_unbiased`、`detoxify_original`、`detoxify_multilingual`、`roberta-hate-speech-dynabench-r4-target`；`toxigen` 因 LangFair 0.8.0 分數反轉的 bug 被拒絕（400）。

### 語言檢查（非英文拒算）

Response 頂層會帶 `languageCheck`：

```json
{
  "languageCheck": {
    "isEnglish": false,
    "method": "ascii_ratio",
    "asciiRatioThreshold": 0.9,
    "checkedResponseCount": 120,
    "nonEnglishResponseCount": 120,
    "nonEnglishResponseIds": [434, 435]
  }
}
```

每筆回覆計算「ASCII 字母 / 全部字母」（`str.isalpha`，標點與數字不計）。只要有一筆低於 0.9，整個 run 拒算：所有選取的 suite（包含 `classification` 與 `recommendation`）都不計算，各自回傳：

```json
{
  "status": "unsupported_language",
  "suite": "toxicity",
  "message": "LangFair toxicity metrics were not computed: ...",
  "languageCheck": { "...": "..." }
}
```

### 補充欄位

所有信賴區間皆為 95% percentile bootstrap（1000 次、固定 seed），格式為 `{"ciLow", "ciHigh", "confidence", "resamples"}`；以 `mean_with_ci` 計算者另含 `value` 與 `n`。`confidenceIntervals` 以指標名稱對應同層 `metrics` 中的值。

| Suite | 欄位 | 說明 |
|---|---|---|
| toxicity | `groupComparison` | `byGroup`（各組指標）、`gap.metrics`（第一組 − 第二組，見 `comparison`）、`gap.confidenceIntervals`、`counts` |
| stereotype | `groupComparison.<category>` | 同上，只針對 Stereotype Classifier 指標 |
| counterfactual | `confidenceIntervals`、`pairCount`、`groups`、`excludedIdenticalPromptPairs` | Sentiment Bias 的 CI 為重抽 pair 後重算 Wasserstein-1 |
| counterfactual | `noiseBaseline` | 同一 prompt 不同次生成之間的相同指標（含自己的 `metrics` / `confidenceIntervals`） |
| recommendation | `metrics` | **已正規化**的 Jaccard / PRAG / SERP（雙向取小值、完全相同 = 1）；LangFair 原始值在 `langfairRawMetrics` |
| recommendation | `confidenceIntervals`、`noiseBaseline`、`excludedIdenticalPromptPairs` | 同 counterfactual |
| classification | `groupGap` | Predicted Positive Rate（有 `expected_label` 時另含 FNR / FPR / FDR / FOR）的有號組間差與 CI |
| classification | `pairedDecisions` | 反事實 pair 決策翻轉率 `Decision Flip Rate`、`positiveOnlyFor`、`mcnemarExactPValue` |

## 10. Persona API

### 取得 Persona Attributes

```http
GET /api/persona/attributes/
```

可選 query parameter：

```http
GET /api/persona/attributes/?pool=persona/datasets/matraix-persona-1m
```

### 取樣 Personas

```http
POST /api/persona/sample/
```

#### Request

```json
{
  "sample_size": 10,
  "selected_attributes": ["gender", "age", "profession"],
  "pool": "persona/datasets/matraix-persona-1m",
  "filters": {},
  "stratify": "gender",
  "seed": 42
}
```

#### Response

```json
{
  "count": 10,
  "pool": "persona/datasets/matraix-persona-1m",
  "selected_attributes": ["gender", "age", "profession"],
  "personas": [],
  "partial": false
}
```

## 11. 建議整合流程

### 方式 A：Backend 主動呼叫 Target LLM

```text
1. POST /evaluations/
2. POST /evaluations/{id}/prompt-populations/generate/
3. GET /prompt-populations/{id}/ 輪詢狀態
4. PUT /evaluations/{id}/llm-config/
5. POST /evaluations/{id}/runs/
6. GET /runs/{run_id}/status/ 輪詢狀態
7. GET /runs/{run_id}/fairness-metrics/
```

### 方式 B：外部機器人執行並回傳結果

```text
1. POST /evaluations/
2. POST /evaluations/{id}/prompt-populations/generate/
3. GET /evaluations/{id}/prompts/
4. 外部機器人自行處理所有 prompts
5. POST /runs/{run_id}/responses/
6. POST /runs/{run_id}/fairness-metrics/
```

方式 B 適合串接多個不同機器人，所有機器人只需要遵循 response API 的格式。
