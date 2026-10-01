# 任務檔範例

建立 Evaluation 時，可以上傳任務檔取代手寫的 Use Case Description。檔案裡的每個 objective 會各自生成一組 prompt。

| 檔案 | Task Type | 內容 |
|---|---|---|
| `customer_service_text_generation.json` | Text Generation | 10 個客服情境：航空、電信、銀行、電子發票、環境部 |
| `customer_service_text_generation.xlsx` | Text Generation | 同上，Excel 版（標題列為「任務目標」） |
| `loan_review_classification.json` | Classification | 5 種貸款與額度審核 |
| `wealth_product_recommendation.json` | Recommendation | 4 種理財需求 |

語言建議選 **Auto detect**，因為 objective 是中文。

## 支援的格式

**JSON**，下面三種寫法都可以：

```json
["任務 A", "任務 B"]
[{"objective": "任務 A"}, {"objective": "任務 B"}]
{"tasks": [{"objective": "任務 A"}]}
```

- 物件的 key 可以是 `objective`、`task`、`target` 或 `description`。
- 最外層的 key 可以是 `tasks`、`objectives` 或 `items`。

**XLSX**：

- 只讀第一個工作表。
- 第一列是標題，會找名稱為 `objective`、`task`、`target`、`description` 或 `任務目標` 的欄位；找不到時就用第一欄。
- 其他欄位（例如這裡的「情境」「備註」）會被忽略，可以當作自己的註記。

## 表單還要填的設定

任務檔只放 objective。下面這些設定要在建立 Evaluation 的表單中另外填寫。

**Classification**（`loan_review_classification.json`）：需要剛好 2 個 label，例如：

| value | name | description |
|---|---|---|
| 0 | reject | 不核准 |
| 1 | approve | 核准 |

**Recommendation**（`wealth_product_recommendation.json`）：候選商品例如：

| name | description |
|---|---|
| 台幣定存 | 保本、固定利率，流動性較低 |
| 投資等級債券基金 | 風險中低，穩定配息 |
| 全球股票型 ETF | 長期成長，波動較大 |
| 高股息 ETF | 定期配息，波動中等 |
| 儲蓄型保險 | 長期保障與儲蓄，提前解約有損失 |
| 貨幣市場基金 | 流動性高、報酬低 |

## 驗證

`apps/evaluations/tests_examples.py` 會檢查這裡的每個檔案都能解析，並模擬前端用 multipart 上傳來建立三種 Evaluation。新增或修改範例後執行：

```bash
uv run python manage.py test apps.evaluations.tests_examples
```
