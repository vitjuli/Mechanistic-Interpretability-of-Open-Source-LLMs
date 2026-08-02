# Colab forward-tasks — как запустить (замена CSD3)

> **2026-08-02.** Для пере-прогонов после находки с моделью используй новый ноутбук
> **`colab/rerun_2026_08.ipynb`** — он делает ровно три задачи (flip-law Δµ с гейтом,
> semantic steering на 8 концептов, дозасев малых c) и начинается с PREFLIGHT-ячейки,
> которая проверяет, что в клоне лежат фиксы и что дампы сняты той же моделью.
> `run_forward_tasks.ipynb` ниже — старый рабочий журнал на 5 задач, оставлен как есть.
> Бандлов теперь два: `make_data_bundle.sh` (задачи A и C) и `make_particles_bundle.sh`
> (задача B).

Пакет в `colab/`: **`run_forward_tasks.ipynb`** (ноутбук на 5 задач) + **`make_data_bundle.sh`** (упаковщик данных) + этот манифест.

## Порядок действий

### 0. ⚠️ Сначала запушить код в GitHub
Ноутбук клонит `vitjuli/Mechanistic-Interpretability-of-Open-Source-LLMs` (ветка `main`). Значит **последний код должен быть на GitHub**, иначе Colab возьмёт старый. Обязательно должны быть запушены:
- `scripts/133_whitening_theorem.py` (патч n_null=50 + family)
- `src/transcoder/activation_functions.py` (фикс Py3.9)
- `scripts/131_delta_sweep_tier2.py`, `scripts/132_flip_law_assembly.py`
- (git-шаги делаешь ты вручную)

### 1. Собрать данные (локально)
```bash
bash colab/make_data_bundle.sh
```
→ создаёт `~/colab_data_bundle.tar.gz` (~0.5 GB: field_dump α/β+grammar, cells_tier2, prompts, кластеры, local_capture). Код в архив НЕ входит (клонится). Модель Qwen3-4B + transcoders качаются в Colab из HF.

### 2. Залить архив в Google Drive
Положи `colab_data_bundle.tar.gz` в корень `MyDrive` (или поправь путь `BUNDLE` в ячейке 0 ноутбука).

### 3. Открыть ноутбук в Colab
- Загрузи `colab/run_forward_tasks.ipynb` в Colab.
- **Runtime → Change runtime type → GPU** (A100/L4/T4; A100 лучше для 131).
- Прогоняй ячейки сверху вниз. Ячейки 0–3 = setup+sanity, дальше — 5 задач, последняя — zip+download выходов.

## Задачи и статус команд

| # | задача | ячейка | команда | статус |
|---|---|---|---|---|
| 1 ★ | **c5 causal** (Table 9 accepted/partial) | Task 1 | `27_cluster_joint_ablation.py --clusters all --device cuda` + `27b` | ⚠ проверь `--behaviour` (имя decay-поведения в твоём config) + args `27b` |
| 2 | **flip-law** 131→132→133 | Task 2 | как в `submit_131_132.sbatch` | ✅ команды точные |
| 3 | **интервенции** | Task 3 | `89_intervention_calculus.py --layers 21 35 ...` | ✅ из j89 (можно поднять calc/flip_targets с smoke-значений) |
| 4 | **w-steering генерации** | Task 4 | `steering_decode_check.py` | ⚠ сначала `--help` (args не зафиксированы), потом раскомментить запуск |
| 5 | **ablation-углы** | Task 5 | `153_rotation_vs_amplitude.py` | ⚠ сначала `--help`, потом запуск |

**★ Приоритет — Task 1 (c5 causal):** снимает «provisional» с Table 9 (финализирует accepted/partial, у Louvain именно c5 демотировал кластеры). Если время/бюджет ограничены — запусти хотя бы её.

## После прогона
Ячейка 6 качает `forward_outputs.tar.gz`. Распакуй в проект (те же пути `data/analysis/runD_v2/...`), затем обнови числа в тезисе/провенансе:
- Table 9 статусы (c5) → снять «provisional».
- flip-law measured-delta collapse (131/132) → обновить строки в RESULTS_PROVENANCE.
- App B.3/B.4 ablation-углы.

## Заметки
- **Дампы уже float32-совместимы**; `field_dump` содержит res+grad (36 слоёв) — 131 и whitening их читают.
- Split-параметры (seed 0, train_frac 0.6, shrink 0.1) зашиты в ячейку Task 2 — менять НЕ надо.
- Если A100 недоступен — L4/T4 тоже пойдут, 131 просто дольше (~час на концепт вместо ~15 мин).
