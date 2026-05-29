# IIA(G) ≈ 0 — диагностика гипотез и сравнение с литературой

**Behaviour**: `physics_decay_type_probe_v2` (538 prompts: 269 α + 269 β)
**Кластеризация**: agglo_coimp_k12 (12 кластеров на 40 фичей)
**Симптом**: IIA_trained ≤ 0.056 для всех 12 кластеров; necessity = 67.6% при joint ablation

Цель: определить, какая из 4 гипотез про IIA ≈ 0 доминирует и сравнить с литературой.

---

## Гипотезы

| # | Гипотеза | Литература |
|---|----------|------------|
| H1 | Неправильная кластеризация (co-imp ≠ causal) | Zhang & Nanda 2024 |
| H2 | Нужен joint patching нескольких кластеров | Chen et al. 2026 (OASR) |
| H3-parallel | Параллельные ensembles активны одновременно | Chen et al. 2026, Anthropic 2025 |
| H3-reactive | Backup pathways активируются после ablation | Wang et al. 2023 (IOI) |
| H4 | Detector vs executor split | Anthropic 2025 (reading/writing) |

---

## Результаты выполненных локальных анализов

### H4 — layer stratification (10 минут, локально)

**Скрипт**: `scripts/diag_h4_layer_stratification.py`
**Данные**: `data/analysis/iia_failure_diagnosis/h4_*.{csv,json}`

| Метрика | Значение | Интерпретация |
|---------|----------|---------------|
| IIA_trained per layer | 0.000–0.056 (max L18) | Все слои даут IIA ≈ 0 |
| Pearson(layer, IIA) | −0.18 | Нет монотонного тренда |
| Pearson(layer, &#124;Δ_shift&#124;) | **+0.64** | Поздние слои сдвигают логиты СИЛЬНЕЕ при патчинге |
| Best layer для shift | L23 (0.69), L24 (0.60) | Executor-кандидаты в L22–L24 |
| Multi-layer C7: L24→L25 | IIA одинаково; shift L24=0.60 > L25=0.29 | L24 — ключевой, L25 уже decoder |

**Verdict для H4**: ⚠ **PARTIAL** — IIA не флипает на поздних слоях, но величина сдвига коррелирует с глубиной слоя. Это означает: поздние слои **производят больше работы** при патчинге, но эффект недостаточен для бинарного flip. Это согласуется с **distributed encoding** (H2) больше, чем с чистым detector/executor разделением.

### H3-parallel — co-activation analysis (4 минуты, локально)

**Скрипт**: `scripts/diag_h3_parallel_coactivation.py`
**Данные**: `data/analysis/iia_failure_diagnosis/h3_*.{csv,json}`

| Метрика | Значение | Интерпретация |
|---------|----------|---------------|
| Mean clusters firing per prompt | **4.33** / 12 | Высокая coactivation |
| Prompts с ≥ 3 кластерами firing | 94.7% | Multi-cluster computation — норма |
| Prompts с ≥ 5 кластерами firing | 42.4% | Часто 5+ кластеров одновременно |
| α-supporting clusters | **5** (C1, C2, C4, C6, C7) | Redundant α-pathways |
| β-supporting clusters | **4** (C0, C3, C5, C8) | Redundant β-pathways |
| Mean inter-cluster corr на α | +0.045 | СЛАБО синхронизированы |
| Mean inter-cluster corr на β | +0.106 | Слабо синхронизированы |
| Top co-firing pair | C7+C8 на 56% prompts | Convergence × β-discriminator |

**Verdict для H3-parallel**: ⚠ **WEAK** — есть redundancy в ролях (5 α + 4 β кластера), но корреляция слабая. Это **не классические OASR-style parallel circuits** (которые требовали бы синхронной активации). Картина больше похожа на **distributed encoding с частично-независимыми pathways** → возврат к H2.

---

## CSD3-эксперименты (готовы к запуску)

### H1 + H2 + H4-detailed — единый sbatch

**Скрипт**: `scripts/53_iia_diagnosis.py` (modes: `h1_circuit`, `h2_pairs`, `h4_layer_split`)
**Job**: `jobs/run_iia_diagnosis_all.sbatch`
**Время**: ~6–8 часов (one GPU)

**H1 — patch top-K circuit features**:
- Top-K = {5, 10, 15, 20, 25, 30} по двум rankings:
  - `top_by_output_edge`: фичи с прямой связью с output
  - `top_by_total_attribution`: Σ &#124;w&#124; по всем edges
- Если IIA(top-30) > 0.3 → проблема была в Louvain кластеризации (**H1 SUPPORTED**)
- Если IIA(top-30) ≈ 0 → distributed, проблема не в кластеризации

**H2 — joint cluster pairs**:
- Pairs: detector+executor (early α + late α, early β + late β)
- Controls: mixed α+β pairs
- Если same-role pairs дают IIA >> single → **H2 SUPPORTED + H4 SUPPORTED одновременно**

**H4 — late vs early layer-only patching**:
- Split layers по {18, 20, 22}
- Patch early-only vs late-only из top-30 circuit features
- Если IIA(late-only) >> IIA(early-only) → executor in late layers

### H3-reactive — отдельный sbatch

**Скрипт**: `scripts/54_h3_reactive_backup.py`
**Job**: `jobs/run_h3_reactive_backup.sbatch`
**Время**: ~6–10 часов (one GPU)

Для каждого кластера: ablate → re-forward → измерить Δ_act на оставшихся 35+ фичах. Backup кандидаты = фичи с систематически положительным Δ_act после ablation.

---

## Прогноз результатов (на основе текущих данных)

Учитывая что:
- Necessity 67.6% (joint ablation работает) → фичи каузально нужны в сумме
- 5 α + 4 β клиника redundancy в ролях
- |Δ_shift| коррелирует со слоем (r=+0.64) → поздние слои executors
- Mean 4.33 кластеров firing per prompt
- IIA на одиночных кластерах ≈ 0

**Наиболее вероятный исход CSD3**:
1. **H1 — частично**: IIA(top-30 circuit) > 0 но < 0.5 — proper feature ranking лучше Louvain, но всё ещё distributed
2. **H2 — главный результат**: detector+executor pairs дадут IIA > 0.4, single ≈ 0
3. **H4 — соответствует**: late-only patching > early-only patching по IIA и shift
4. **H3-reactive — слабо**: backup найдётся, но эффект небольшой (parallel structure из H3-parallel уже работает в ОБЫЧНЫХ прогонах, reactive backup менее нужен)

**Главный тезис для диссертации** (если прогноз подтвердится):
> IIA(G) ≈ 0 для одиночного кластера G — следствие *distributed sufficient cause*: каузальная переменная V_h реализована редундантно как параллельная сборка ≥ 2 (по типу — детектор + исполнитель) механизмов. Это не отрицает Condition II — она операционализуется через `Nec(G)` (joint ablation) или через `IIA(G_pair)` (с пары детектор+исполнитель). Прецеденты в литературе: backup heads (Wang 2023), redundant SAE features (Marks 2024), parallel sheaves (OASR 2026), reading vs writing components (Anthropic 2025).

---

## Сравнение с литературой (matrix)

| Литература | H1 | H2 | H3-parallel | H3-reactive | H4 |
|------------|----|----|-------------|-------------|----|
| Wang et al. 2023 (IOI, ICLR) | — | — | partial | **★main** | — |
| Marks et al. 2024 (SFC, ICLR 2025) | partial | **★main** | — | — | partial |
| Geiger et al. 2024 (JMLR, DAS) | partial | — | — | — | — |
| Anthropic 2025 (Bio of LLM) | — | partial | partial | partial | **★main** |
| Chen et al. 2026 (OASR) | — | partial | **★main** | — | — |
| Conmy et al. 2024 (ACDC, ICLR) | partial | partial | — | partial | — |
| Heimersheim & Nanda 2024 | partial | — | — | — | — |
| Lieberum et al. 2023 | — | — | partial | partial | — |
| Zhang & Nanda 2024 | **★main** | — | — | — | — |
| Wu et al. 2025 (Non-Linear Dilemma) | — | — | — | — | — |

**Главный итог**: ни одна работа не утверждает, что IIA ≈ 0 = провал интерпретации. Все они подтверждают, что **distributed/redundant encoding — норма** в LLM > 4B параметров, и интерпретация переходит на circuit-level (joint patching, necessity, DAS).

---

## Следующие шаги

1. `git add scripts/diag_h4_layer_stratification.py scripts/diag_h3_parallel_coactivation.py scripts/53_iia_diagnosis.py scripts/54_h3_reactive_backup.py jobs/run_iia_diagnosis_all.sbatch jobs/run_h3_reactive_backup.sbatch data/analysis/iia_failure_diagnosis/`
2. `git commit -m "Add IIA failure diagnosis: H1-H4 hypotheses tests"`
3. `git push`
4. На CSD3: `git pull && sbatch jobs/run_iia_diagnosis_all.sbatch && sbatch jobs/run_h3_reactive_backup.sbatch`
5. После завершения: sync результаты, обновить SYNTHESIS.md финальной таблицей
6. Дописать в `thesis_1_2.md` секцию "Why IIA(G) ≈ 0: empirical investigation" со ссылками на эти эксперименты
