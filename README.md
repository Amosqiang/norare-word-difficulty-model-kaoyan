# norare-word-difficulty-model-kaoyan

这是一个基于 NoRaRe CLDF 1.1 训练的单词难度模型，并支持将考研词汇频率（`words.json`）融合进训练标签。

## 环境要求
- Python 3.9+。
- 依赖：`numpy`, `pandas`, `scikit-learn`, `joblib`。

安装依赖：
```bash
python3 -m pip install --user numpy pandas scikit-learn joblib
```

## 训练 / 重训练
1) 将 `Norare CLDF 1.1.zip` 放到项目根目录。
2) （可选）将 `words.json` 放到项目根目录用于融合词频。
   - `words.json` 来自 NETEMVocabulary，为考研词汇频率表。
3) 运行：
```bash
python3 train_difficulty_model_from_norare.py   --norare_zip "Norare CLDF 1.1.zip"   --out_dir .   --work_dir norare_train_model_work   --words_json words.json   --freq_weight 0.35
```

## 参数说明
- `freq_weight`：频率难度占比（0..1），0 表示只用 NoRaRe 的 PCA 标签。
- `freq_clip_low` / `freq_clip_high`：`log1p(freq)` 的分位数裁剪（默认 1/99）。
- 频率难度映射：`log1p(freq)` -> p1/p99 裁剪 -> 0..100（高=更难）。
- 若不想融合词频，可设置 `--freq_weight 0` 或 `--words_json ""`。

## 训练输出
- `difficulty_model_any_word.joblib`（模型）
- `norare_training_database_en.csv`（训练数据库）
- `model_manifest.json`（训练元信息）


## 字段含义
`words_with_difficulty.json` 中新增的字段：
- `difficulty_freq_0_100`：基于 `words.json` 频率的难度分数（log1p + 分位裁剪 -> 0..100，数值越大越难）。
- `difficulty_norare_pred_0_100`：NoRaRe 模型预测的难度分数（0..100）。
- `difficulty_0_100`：融合后的最终难度分数（`freq_weight` + `norare_weight` 加权）。

`norare_training_database_en.csv` 中主要字段说明：
- `word` / `form_norm`：原始词与小写规范化词形。
- `aoa_mean` / `prevalence` / `known_pct` / `freq_log` / `cd` / `concreteness`：NoRaRe 词汇规范指标。
- `*_source_varid`：对应指标的来源变量 ID（来自 NoRaRe）。
- `difficulty_label_0_100_pca`：仅由 NoRaRe 指标 PCA 得到的难度标签（0..100）。
- `difficulty_label_0_100`：最终训练标签（可包含词频融合）。
- `frequency` / `freq_log1p` / `difficulty_freq_0_100` / `freq_weight_used`：仅在启用词频融合时出现。

## 预测（可选）
```bash
python3 predict_any_word.py --word "abandon" --model difficulty_model_any_word.joblib
```

## 示例运行参数（来自 `model_manifest.json`）
- `freq_weight`: 0.35
- `clip_low_percentile`: 1.0
- `clip_high_percentile`: 99.0
- `log_transform`: log1p
- `p1`: 0.0
- `p99`: 7.980793305568741
- `matched_rows`: 1874

## 数据来源与许可
- NoRaRe CLDF 1.1（Concepticon / NoRaRe contributors）
  - 许可证：Creative Commons Attribution 4.0 International (CC BY 4.0)
  - 来源：https://github.com/concepticon/norare-cldf
- 考研词汇频率表（NETEMVocabulary）
  - 来源：https://github.com/exam-data/NETEMVocabulary

训练标签由 NoRaRe 中的多项英文词汇规范数据衍生而来，包括：
- Kuperman et al. (2012) Age of Acquisition (AoA)
- Brysbaert et al. (2019) Word Prevalence / Known percentage
- Van Heuven et al. (2014) SUBTLEX-based frequency (log) 和 contextual diversity (CD)
- Brysbaert et al. (2014) Concreteness

难度分数为派生值（基于 PCA 的潜在难度），并非 NoRaRe 的原始字段。

## 备注 / Troubleshooting
- 如果提示缺少依赖，请先安装上面的依赖。
- 训练过程中可能出现数值警告（PCA/Ridge），但文件仍会正常生成。

## 生成 words.json 的预测结果
训练完成后，可批量为 `words.json` 生成难度字段：

```bash
python3 add_difficulty_to_words_json.py   --words_json words.json   --model difficulty_model_any_word.joblib   --out_json words_with_difficulty.json   --freq_weight 0.7   --norare_weight 0.3
```

输出文件：`words_with_difficulty.json`。
若只想用模型预测（不融合词频），可设置 `--freq_weight 0 --norare_weight 1`。

