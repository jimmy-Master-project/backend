---
tags:
  - LLM-Fairness
  - LangFair
  - Evaluation-Metrics
created: 2026-08-17
source: langfair==0.8.0 source code inspection
---

# LangFair 五個公平性評估向度與指標

> LangFair 是用於評估 LLM fairness / bias 的開源套件。以下整理 `langfair==0.8.0` 中五個主要向度、內建指標與計算方式。

## 總覽

| 向度 | 指標 | ＠
|---|---|
| Counterfactual Fairness 反事實公平性 | Cosine Similarity、RougeL Similarity、Bleu Similarity、Sentiment Bias |
| Toxicity 毒性 | Toxic Fraction、Expected Maximum Toxicity、Toxicity Probability |
| Stereotype 刻板印象 | Stereotype Association、Cooccurrence Bias、Stereotype Classifier 指標群 |
| Recommendation Fairness 推薦公平性 | Jaccard、SERP、PRAG、SNSR、SNSV |
| Classification Fairness 分類公平性 | FNRP、FORP、FPRP、FDRP、PPRP |

---

## 1. Counterfactual Fairness 反事實公平性

比較兩組只改變 protected attribute 的 LLM 回答，例如 gender 或 race 替換後的輸出。

### Cosine Similarity

將兩段文字轉成 sentence embedding 後計算 cosine similarity：

$$
\text{cosine}(x,y)=\frac{e_x \cdot e_y}{\|e_x\|\|e_y\|}
$$

越高代表反事實替換後語意越相近。

### RougeL Similarity

使用 ROUGE-L F-measure，比較兩段文字的 longest common subsequence 相似度。

越高代表文字序列越相似。

### Bleu Similarity

LangFair 對每組 counterfactual pair 計算雙向 BLEU，並取較小值：

$$
\min(\text{BLEU}(t_1 \rightarrow t_2),\text{BLEU}(t_2 \rightarrow t_1))
$$

越高代表文字越相似。

### Sentiment Bias

比較兩組 counterfactual responses 的 sentiment score 差異。

#### Strong parity，預設

使用 Wasserstein-1 distance：

$$
W_1=\frac{1}{n}\sum_i |s^{(1)}_{(i)}-s^{(2)}_{(i)}|
$$

其中兩組 sentiment scores 會先排序。

#### Weak parity

先用 threshold 二值化：

$$
\hat{y}=1[s>\tau]
$$

再計算兩組 positive rate 差：

$$
\mathbb{E}[\hat{y}_1]-\mathbb{E}[\hat{y}_2]
$$

---

## 2. Toxicity 毒性

LangFair 支援多種 toxicity classifier，例如 `detoxify_unbiased`、`detoxify_original`、`detoxify_multilingual`、`roberta-hate-speech-dynabench-r4-target`、`toxigen`。

若使用多個 classifier，LangFair 對每個 response 取最高 toxicity score：

$$
s_i=\max_c s_{i,c}
$$

### Toxic Fraction

毒性分數超過 threshold 的比例：

$$
\frac{1}{n}\sum_i 1[s_i>\tau]
$$

### Expected Maximum Toxicity

同一 prompt 多個回答中的最大毒性分數：

$$
\max_i s_i
$$

若有多個 prompts，先對每個 prompt 計算 max，再取平均。

### Toxicity Probability

同一 prompt 是否至少有一個回答超過 threshold：

$$
1[\max_i s_i>\tau]
$$

若有多個 prompts，對每個 prompt 計算後再取平均。

---

## 3. Stereotype 刻板印象

### 3.1 Stereotype Association

計算 demographic words 與 target stereotype words 的共現偏差。

對每個 target word $t$，統計它與各 demographic group $g$ 的共現次數 $c_g$。先除以該 group word list 長度：

$$
c'_g=\frac{c_g}{|\text{wordlist}_g|}
$$

再正規化為機率分布：

$$
p_g=\frac{c'_g}{\sum_g c'_g}
$$

與 uniform distribution 比較：

$$
u_g=\frac{1}{G}
$$

最後計算 total variation distance：

$$
\text{TV}=\frac{1}{2}\sum_g |p_g-u_g|
$$

越高代表 demographic group 與 stereotype target words 的關聯越不均衡。

### 3.2 Cooccurrence Bias, COBS

使用距離加權共現計算兩組 demographic words 的偏差。權重為：

$$
\beta^d
$$

其中 $d$ 是 target word 與 protected attribute word 的 token distance，預設 $\beta=0.95$。

對每個 target word $w$：

$$
\text{COBS}(w)=\left|\log_{10}\frac{P(w|g_1)/P(g_1)}{P(w|g_2)/P(g_2)}\right|
$$

可回傳每個 word 的 COBS，或平均 COBS。越高代表共現偏差越大。

### 3.3 Stereotype Classifier

使用 HuggingFace 模型：

```text
wu981526092/Sentence-Level-Stereotype-Detector
```

支援類別：gender、race、profession、religion。

基於 classifier score 後，計算：

| 指標 | 計算方式 |
|---|---|
| Stereotype Fraction | 分數超過 threshold 的比例 |
| Expected Maximum Stereotype | 同 prompt 多個回答中的最大 stereotype score |
| Stereotype Probability | 同 prompt 是否至少一個回答超過 threshold |

公式與 toxicity 指標相同。

---

## 4. Recommendation Fairness 推薦公平性

用於比較不同 protected attribute group 下，LLM 產生的推薦清單是否一致。

### Jaccard

對兩個推薦清單 $A,B$：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

越高代表推薦項目越相似。

### SERP

LangFair 原始碼中的計算：

$$
\text{SERP}=\frac{0.5\sum_{i:item_i\in B}(K-i+1)}{K(K+1)}
$$

其中 $K$ 是推薦清單長度，$i$ 是 list1 中的位置，**從 0 開始**（原始碼 `enumerate(list1)`），所以第一個項目的權重是 $K+1$。越前面的項目若也出現在另一清單，貢獻越大。

> 注意：兩清單完全相同時，SERP $=\frac{K+3}{4(K+1)}$（K=3 時為 0.375），不會到 1。

### PRAG

比較 list1 中 item pairs 的相對順序是否在 list2 中維持一致：

$$
\text{PRAG}=\frac{\text{matches}}{K(K+1)}
$$

其中 matches 是 list1 中每對 item 在 list2 中維持相對順序的數量。

> 注意：兩清單完全相同時，PRAG $=\frac{K-1}{2(K+1)}$（K=3 時為 0.25），不會到 1；K=1 時直接回傳是否相同（0/1）。
>
> LangFair 的 `evaluate_pairwise` 內部 `_pairwise_calculations` 寫成 `min(m(a,b), m(a,b))`，原意應是雙向取小值，但兩次參數順序相同，所以 SERP/PRAG 實際上是單向的。

### SNSR / SNSV

在 `evaluate_against_neutral` 中，先計算每個 protected group 相對 neutral list 的 metric value $m_g$。

$$
\text{SNSR}=\max_g m_g-\min_g m_g
$$

$$
\text{SNSV}=\text{std}(m_g)
$$

SNSR / SNSV 越低代表各群體間推薦相似度差異越小。

---

## 5. Classification Fairness 分類公平性

用於二元分類結果。

輸入：

- `groups`：protected group indicator，必須剛好兩組
- `y_pred`：binary prediction，0/1
- `y_true`：binary label，0/1

LangFair confusion matrix 定義為：

$$
\begin{bmatrix}
TN & FP \\
FN & TP
\end{bmatrix}
$$

### 指標與公式

#### False Negative Rate Parity, FNRP

$$
FNR=\frac{FN}{FN+TP}
$$

#### False Positive Rate Parity, FPRP

$$
FPR=\frac{FP}{FP+TN}
$$

#### False Discovery Rate Parity, FDRP

$$
FDR=\frac{FP}{FP+TP}
$$

#### False Omission Rate Parity, FORP

$$
FOR=\frac{FN}{FN+TN}
$$

#### Predicted Prevalence Rate Parity, PPRP

$$
PPR=\mathbb{E}[y_{pred}]
$$

也就是 predicted positive 的比例。

### Parity 計算

預設 `ratio=False`，使用絕對差：

$$
\text{parity}=|r_1-r_2|
$$

若 `ratio=True`：

$$
\text{parity}=\frac{r_1}{r_2}
$$

difference 越接近 0 越公平；ratio 越接近 1 越公平。

---

## 快速解讀

| 指標類型 | 公平時的方向 |
|---|---|
| similarity 類，如 Cosine、RougeL、BLEU、Jaccard、SERP、PRAG | 越高通常越好 |
| bias / toxicity / stereotype fraction 類 | 越低通常越好 |
| parity difference 類，如 FNRP、FPRP、FDRP、FORP、PPRP | 越接近 0 越公平 |
| parity ratio 類 | 越接近 1 越公平 |

---

## 本系統在 LangFair 之上的修正與補充（`apps/fairness/services`）

| 項目 | 做法 | 原因 |
|---|---|---|
| 非英文拒算 | 每筆回覆計算「ASCII 字母 / 全部字母」（`str.isalpha`，中文字也算字母；標點、數字不計），低於 0.9 視為非英文。只要有任一筆非英文，整個 run 拒算：所有選取的 suite 都回傳 `status: unsupported_language` | detoxify_unbiased、stereotype classifier 與詞表、VADER、rouge_score 分詞、all-MiniLM-L6-v2 都只支援英文。實測中文下 VADER 恆為 0、RougeL 只比對 ASCII 片段（中文內容完全不同也可得 1.0）、MiniLM 對無關中文句子 cosine 也有 0.5–0.7 |
| SERP / PRAG 正規化 | 每對清單取 $\min(m(a,b), m(b,a))$，再除以「同長度清單與自身比較」的分數，使完全相同 = 1；LangFair 原始值保留在 `langfairRawMetrics` | 原始值上限隨 K 改變，不同 run 無法比較，也與「越高越好、1 為完全一致」的直覺不符 |
| 依組別拆開 | toxicity 與 stereotype classifier 指標依 protected group 各算一次，回報 `groupComparison.byGroup` 與組間差 `gap`（第一組 − 第二組） | LangFair 把所有回覆混在一起算，只能看整體安全性，看不出組間差異 |
| 信賴區間 | 95% percentile bootstrap（1000 次、固定 seed）：counterfactual 對 pair 重抽、組間差在各組內對 prompt / 回覆重抽、Sentiment Bias 重抽 pair 後重算 Wasserstein-1 | 單次結果的小差距常落在隨機誤差內 |
| 雜訊基準 `noiseBaseline` | counterfactual / recommendation 另外計算「同一 prompt 不同次生成之間」的相同指標 | 跨組相似度要明顯低於（CI 不重疊）同組雜訊基準，才代表有偏差 |
| 決策翻轉率 `pairedDecisions` | 分類任務中，同一對反事實 prompt 只換 protected attribute 後決策改變的比例；並以精確 McNemar 檢定（不一致 pair 的二項檢定）判斷翻轉是否偏向某一組 | 成對比較比組別層級的 PPRP 敏感得多 |
| 組間差率 `groupGap` | PPR（及有 `expected_label` 時的 FNR/FPR/FDR/FOR）的有號差與 CI | LangFair 只回報絕對差，沒有不確定性 |
| 排除替換失敗的 pair | original 與 counterfactual prompt 文字完全相同的 pair 不納入（回報 `excludedIdenticalPromptPairs`） | 這類 pair 的差異只反映生成隨機性 |
| 停用 toxigen | `toxicity.classifiers` 只接受 detoxify_unbiased / original / multilingual 與 roberta-hate-speech | LangFair 的 toxigen 分支以 `r["label"][-1] == 1` 比較字串與整數，永遠為 False，分數被反轉成 1 − p |
