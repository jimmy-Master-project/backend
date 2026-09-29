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

其中 $K$ 是推薦清單長度，$i$ 是 list1 中的位置。越前面的項目若也出現在另一清單，貢獻越大。

### PRAG

比較 list1 中 item pairs 的相對順序是否在 list2 中維持一致：

$$
\text{PRAG}=\frac{\text{matches}}{K(K+1)}
$$

其中 matches 是 list1 中每對 item 在 list2 中維持相對順序的數量。

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
