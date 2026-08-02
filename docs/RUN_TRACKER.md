# Run Tracker — что где считается (CSD3)

_Обновлено: 2026-08-02. Живой документ — обновляем по мере завершения джобов._

## 🔴 ПЕРЕ-ПРОГОН — flip-law Δµ (131→132) считался на ЧУЖОЙ модели
**Установлено 2026-08-02.** Джоба 131+132+133 отработала 11–12 июля **на Colab** (`colab/run_task2_fliplaw.sh`, копия на Drive `MyDrive/task2_out/`), выходы синкнуты локально 12 июля 14:33. Но `scripts/131_delta_sweep_tier2.py` по умолчанию брал **`Qwen/Qwen3-4B-Base`**, тогда как дампы 119 и свипы 122 — **`Qwen/Qwen3-4B`** (записано в `field_dump/meta.npz`). Continuity-гейт по usage: max |Δflip| = **0.41** (α/β) и **0.51** (grammar), за ±0.03 — 19/35 и 15/42 ячеек.
- **Грязное:** строки `delta` в `flip_law_{pool,train}/flip_law_master.csv`, сводные в `numbers_for_thesis.json` (realized_disp MAE 0.096/0.112 против 0.023/0.037 до Δµ), `heldout_F.json`, `cells_tier2_delta.csv` целиком.
- **Чистое (не трогать):** usage / w_res / random / shuffled (в 132 `drop_duplicates` оставляет запись из `cells_tier2.csv`), `ignition_map` (только usage), `predicted_*_linear` = 0.028/0.030, весь whitening 133 (модель не грузится).
- **Починено в коде:** дефолт 131 → `Qwen/Qwen3-4B` + assert против `meta.npz`; явный `--model_name` в `colab/run_task2_fliplaw.sh` и `jobs/submit_131_132.sbatch`; 133 в colab-раннере выключен (`RUN_133=1` чтобы вернуть).
- **Гейт:** `scripts/check_continuity_131_122.py` (`--self_test` проходит), вшит в colab-раннер между 131 и 132 — при провале `set -e` роняет прогон, так что 132 на плохих ячейках не соберётся.
- **Запуск:** `bash colab/run_task2_fliplaw.sh` (Colab, ~1–2 ч). CSD3-sbatch починен, но не отправляем.

## 🔴 ПЕРЕ-ПРОГОН — semantic steering (demo-свип): не тот контраст + не та модель
**Установлено 2026-08-02.** Бандл `data/analysis/steering_named/semantic_L22_L24_L35/` (прогон 11–12 июля, Colab).
- **Дефект 1 (частицы):** контраст брался из per-prompt поля `incorrect_answer`, а это произвольный дистрактор v2-корпуса, не партнёр пары. Аудит `scripts/audit_semantic_steering_labels.py` → `label_audit.csv`: **64/128 строк годны** (neutron_vs_proton / alpha_beta / grammar — целиком, electron_vs_photon и electron_vs_proton — половина, три пары — ноль).
- **Дефект 2 (все строки):** `steering_decode_check.py` по умолчанию грузил `Qwen3-4B-Base` при дампах `Qwen3-4B`.
- **Знак пуша НЕ пострадал** (`class_a` из дампа для пар, `y_canonical` для α/β и grammar).
- **Починено:** контраст = второй класс самого корпуса, `--model` по умолчанию `Qwen/Qwen3-4B` + assert, `colab/run_semantic_steering.sh` пинит чекпоинт.
- **Пере-гонять все 8 концептов** (не только 6 пар — дефект 2 задел и α/β с grammar), ~2–4 ч Colab.

## ⏳ Дозасев малых c (закрывает `<<demo>>` в §5.2 статьи)
После фикса 122 (delta в dirs по умолчанию, грид `{1/16…32}`, `--c_anchor`): tier-2 только на новых значениях `--t2_c_grid 0.0625 0.125 0.25` на тех же слоях, α/β и grammar, все направления. Сетка-надмножество, старые ячейки не аннулируются, 132 сливает файлы. Даёт linear ∧ informative ячейки на поздних слоях.

## ✅ Переезд на SL2 (2026-06-27)
Раньше всё стояло на `FERGUSSON-SL3-GPU --qos=gpu2` (бесплатный SL3, низкий приоритет → 10ч+ PD). Все sbatch'и переведены на **`MPHIL-DIS-SL2-GPU --partition=ampere`** (без qos) — основная MPhil-аллокация, высокий приоритет. Старые SL3-джобы отменены (`scancel -u iv294`), всё пере-submit на SL2.
- SL2 GPU бюджет: ~2604 ч свободно (хватает).
- Header теперь как у рабочих SL2-джобов: `-A MPHIL-DIS-SL2-GPU`, `-p ampere`, `--gres=gpu:1`, БЕЗ `--qos`.

---

## ✅/🔴 ОТРАБОТАЛО 12.07 НА COLAB, НО ТРЕБУЕТ ПЕРЕ-ПРОГОНА — flip-law: досвип Δµ + сборка закона (131 + 132)
> Секция ниже — исходная постановка от 2026-07-06, оставлена как контекст (kill-критерии, split-параметры, continuity-правило). Фактический статус и что именно грязное — в красной секции вверху файла. Главное: continuity cross-check, описанный здесь как «главный детектор несовпадения конвенций», сработал ровно по назначению и поймал подмену модели.

**Задача (2026-07-06):** замкнуть flip-law — **measured identity-line collapse** (α/β + grammar), класс-условная F, intact-conditioned, честный MAE на информативных ячейках. Поднимает Figure 1 (§3.1.2) до параметр-свободного закона `flip_rate = F_{|m|}(x)`, где `x = c·σ·|cos(v,u)|·‖∇m‖`.
- **Скрипты готовы (в проекте, оба self-test'а проходят локально):** `scripts/131_delta_sweep_tier2.py` (GPU — досвип δ+usage), `scripts/132_flip_law_assembly.py` (CPU — сборка). Джоб: `jobs/submit_131_132.sbatch` (**полностью готов, ручных правок НЕ нужно**: SL2, `venv`, corpus `data/prompts/`, cd в hpc-work).
- **Запуск (как CSD3 встанет):** `git pull && sbatch jobs/submit_131_132.sbatch` (одна джоба: 131 GPU ~1–2ч + 132 CPU, оба концепта, F в двух дисциплинах pool/train).
- **Split-параметры ПРОВЕРЕНЫ = дефолты 122** (`jobs/B1_*_phase3.sbatch` НЕ передают split → seed=0, frac=0.6, shrink=0.1; `σ=std(H_train@w_res)` зависит от них). Вписаны в sbatch. **НЕ менять.**
- **Continuity cross-check (главный детектор несовпадения конвенций):** `usage` из 131 vs 122 `flip_norm` **±0.03** — если разошлось, split/shrink не те, **predicted-ветку НЕ читать**, сначала чинить.
- **Kill-критерии** (из `numbers_for_thesis.json` на каждый концепт): `realized_disp.mae` ≤0.08 (пол шума ~0.02–0.04); `|realized_disp.bias|`<0.02; `heldout_F.degradation`<0.05; delta ложится на identity-line; predicted на мусор-градиентах ДОЛЖЕН падать (MAE 0.34–0.40 — нетавтологичность).
- **Покрытие measured:** reading/use/random/shuffled × 7 сил (0.5–32) × 5 слоёв (16,22,23,24,35) × 2 концепта; **Δµ досвипывается в 131** (в 122 его не было). Локально уже прошло: predicted honest MAE=**0.072** на info-ячейках, класс-условная F **без расщепления** (α/β перемешаны на identity-line), знаковый фикс применён (`s_y·⟨g,v⟩`, baseline-correct пул).
- **Выход → тезис:** `data/analysis/runD_v2/*/flip_law_*/numbers_for_thesis.json` (ключи 1:1 к плейсхолдерам текста). Полный порядок/чек-лист: `docs/flip_law_RUN_ORDER.md`.
- **Whitening-теорема (Прил. F, `scripts/133_whitening_theorem.py`, патч-версия с null-контролем)** — CPU-only, в sbatch (после 132), но **УЖЕ ПОСЧИТАНА ЛОКАЛЬНО на обоих концептах** (~4 мин α/β, ~8 мин grammar). Числа: `data/analysis/runD_v2/*/whitening/whitening_numbers.json`. **На CSD3 НЕ ждёт** — GPU только для flip-law (131).
  - **Ядро (SOLID оба концепта):** rayleigh_pctl(u)=**98.4/98.7%** (механизм F.4); point-prediction cos(w,u) MAE **0.039/0.044**; §2.5.2 привязана к оценщику.
  - **R2 (точная ошибка):** `eigen_residual_effective`=**0.038/0.048** (=\|⟨ŵ,r⟩\|/σ_u², ≈MAE) — спектральная разведённость: КБ-граница ‖r‖/σ_u²=4.4 пуста, но точная ошибка крошечная, т.к. ŵ (малодисп.) ⊥ r (высокодисп.).
  - **R1 (маргин-тождество F.3):** ratio≈**1.16** у readout (L35) → держится у выхода; в зоне ≈**20** → избыток от гетерогенности градиента (тот же CV(drive)≈0.62 из flip-law → **сквозной параметр, 3-е появление**). Каузальную стрелку «флуктуации маржина → u высокодисперсна» — чисто только у выхода; в зоне: «u в высокодисп. подпространстве (эмпирика) + сцеплена с маргином» (сцепленность = контрфактуал).
  - **R3 (контрфактуал, развилка разрешена):** excess_over_null=**0.08–0.12 (α/β) / 0.05–0.06 (grammar)**, measured **> null p95** на обоих → эффект **реален**, но сырой рост cos→0.13–0.25 в основном **small-sample drift** (null median 0.08–0.14, d_eff~13). Формулировать как **excess-over-null**, не абсолютный рост; excess > первопорядкового предсказания (0.006–0.011) → вращение сильнее тождества (вторичный результат). Kill-крит. (iv) закрыт.
- **§2.5.2 приведён к коду (2026-07-06):** был Ledoit-Wolf identity-shrinkage, стал **диагональный shrinkage $s=0.1$** ($\Sigma_s=(1-s)S_W+s\,\mathrm{diag}\,S_W$) — как в `fisher_axis` (122/131/132/133). `ledoit2004` убран из цитирования. ⚠ Побочно: `plot_3_1_3_geometry.py` (фигура cos_by_layer) всё ещё считает identity-shrinkage — cos-числа совпадают (~0.02, подтверждено 133), но для полной консистентности фигуру можно перегенерить с диагональным (низкий приоритет).
- **F.7 (Park–Choe–Veitch связь)** — написана из знаний без live-проверки, **сверить их точную конструкцию перед сдачей** (causal inner product / whitening).

## ⏳ ОЖИДАЕТ CSD3 — steering: компонента флипа (длина/вращение) по вектору×концепту
**Задача (зафиксирована 2026-06-28):** для КАЖДОГО концепта × КАЖДОГО вектора (δ/u/w_res) — при смене ответа какая компонента больше: **вращение (⊥state)** или **изменение длины (∥state)**. Плюс **связь с γ и между осями**.
- **Скрипт готов:** `scripts/cos_wres_u.py` (numpy, login-узел, БЕЗ GPU). Авто-обнаружение всех `field_dump_119` (α/β, grammar, 6 пар).
- **Считает:** `cos(w_res,u)` (механизм инертности) + `cos(δ/u/w_res, state)` (длина∥/вращение⊥) + **`cos(δ/u/w_res, γ)`** (γ ≈ grad последнего слоя = readout-контраст; смотрим, флипает ли в сторону γ) + `cos(u,δ)`.
- **Запуск (как CSD3 встанет):** `git pull && python scripts/cos_wres_u.py` → `data/analysis/27c/axis_geometry_by_concept.csv`.
- **Нужно для полноты:** дампы `field_dump_119` лежат ТОЛЬКО на CSD3 (локально нет .npy).
- **Уже готово локально (не ждёт CSD3):** inter-axis косинусы (`data/analysis/27c/inter_axis_cosines.csv`: cos δ-u≈0.25 везде; cos δ-wres α/β 0.13→particles 0.5; cos u-wres α/β 0.02→particles 0.15) + rot_frac-траектории (realized_*.csv, все оси rotation-доминированы 0.89-0.99).
- **Идея (от пользователя):** с γ и другими осями — вдруг флип не только меняет компоненту, но и вращается в КОНКРЕТНУЮ сторону (к γ/к другому классу) → увидим направленную связь. Если статика (cos) покажет намёк — добить ДИНАМИКОЙ: GPU-скрипт steer-axis + capture state → проекция Δstate на {γ,u,wres,δ} (отдельная задача, тяжелее).

## Прочая очередь на CSD3 (накопилось)
1. `git pull` → `python scripts/cos_wres_u.py` (↑ эта задача, приоритет, numpy).
2. `python scripts/simplex_check.py` (если не прогнан) — u-симплекс K=4.
3. пере-прогон job2 (`run_27c_jobs.sbatch` task3) с патчем движка → `ddelta_cos_{gamma,u,wres}` (знаковые проекции записи кластеров).
4. `sbatch jobs/run_decode_check.sbatch` — форма δ/w_res-флипа (генерация, w_res добавлен).
5. `sbatch jobs/run_neighbor_matrix.sbatch` — «любимый партнёр» частиц.
6. `sbatch jobs/run_particle_27c.sbatch` — particle-кластеры × геометрия.
7. забрать part2 (C4–C7) если досчитался → `recompute_geometry_normfree.py`.

## 🔴 ОЖИДАЕТ — пересчёт кластеров на HAC (метод тезиса ≠ текущие данные)
**Задача (зафиксирована 2026-06-30):** все кластерные таблицы §3.5/§3.7/Appendix B сейчас на **СТАРОЙ co-importance Louvain партиции**, а тезис (thesis_1_1 стр.10/417, thesis_1_2 стр.39, §2.10) коммитит **average-linkage agglomerative (HAC) на co-importance Jaccard**. Тезис сам помечает это «⏳ re-run на agglo+RunD» (thesis_1_2 стр.315/371). Цифры провизорные.

**Доказательная база (проверено локально):**
- Каноническая membership = `data/results/clustering/cluster_membership_ch5.csv` (**40 фич, 11 кластеров C0-C10, со слоями, final_status**). ARI с `coimp_louvain`=**1.000**, с HAC=0.14-0.36 → **данные = Louvain**.
- Валидация (канон) `data/results/cluster_semantics/final_cluster_validation_table.csv` (11): **6 accepted / 3 partial / 2 descriptive** (= `docs/final_latent_state_validation.md`).
- ⚠️ ЕСТЬ ВТОРОЙ файл `data/analysis/runB/cluster_semantics/final_cluster_validation_table.csv` (**12 кластеров, 11 accepted/1 partial, другие lift**) — НЕ канон, но §3.5 Table 8 lift я ошибочно взял оттуда.

**Что сделать на пересчёте:**
1. Перезапустить кластер-уровневые результаты (lift, fire α/β, углы ablation, validation) на **HAC-партиции** (колонки `hac_average_*` в `cluster_labels.csv`, либо заново). Нужна модель → CSD3 (firing/geometry).
2. **Свести §3.5 Table 8 (сейчас runB-12) и B.8 (results-11) на ОДНУ партицию** — рекомендую `cluster_membership_ch5.csv` (11, его реально используют firing+geometry).
3. **Решить k:** thesis_1_1 = **k=16**, thesis_1_2 = **k=12** — главы противоречат. Выбрать одно.
4. ⚠️ **ARI(Louvain,HAC)=0.14-0.36** → состав кластеров при HAC может **заметно измениться** (не «без серьёзных различий»). Перепроверить cue-метки, lift, fire, acceptance.

**Затронутые куски текста (обновить после пересчёта):** §2.10 (метод — оставить HAC), §3.5 Table 8 (lift+статусы+состав), §3.7.2/3.7.3 (последовательность), Appendix B.3/B.4/B.6/B.7/B.8 (углы/membership/acceptance). Сейчас везде стоят провизорные Louvain-числа + пометка.

**✅ HAC-построение УЖЕ ЕСТЬ (найдено 2026-06-30):** `data/analysis/runB_agglo/cluster_semantics/` — **12 HAC-кластеров** с membership (`cluster_feature_summary.csv`: feat→cluster→layer→role), cue enrichment (`cluster_weighted_enrichment.csv`), coherence/reuse/stability. HAC ≠ Louvain (хаб дроблён: HAC-C5 14фич L14/16/17/22 + C11 L23; L19/L21 раздельно C9/C10). HAC top-cue: C8 charge L10, C1 lepton L11, C5 n→p hub, C3 output L24-25.
**Пересчёт в основном ЛОКАЛЬНЫЙ (не CSD3):**
- fire α/β для HAC → `activation_matrix.npy` (227×538) + HAC membership = среднее по классу. ЛОКАЛЬНО.
- acceptance c1-c4,c6 → из существующих runB_agglo (coherence/reuse/stability/enrichment). ЛОКАЛЬНО.
- ТОЛЬКО **c5 causal (IR/SFR, joint ablation)** + **ablation-углы (B.3/B.4)** → нужен CSD3/модель.
**Нет для HAC:** final c1-c6 acceptance table (final_status), fire α/β, ablation-углы — пересчитать (см. выше, большинство локально).

**✅ ЛОКАЛЬНЫЙ HAC k=12 пересчёт СДЕЛАН (2026-06-30):**
- fire α/β (формула 23) → `data/analysis/27c/hac_k12_fire.csv` (11/12 кластеров; C6/L18 фича не в activation_matrix). Все balanced → «physical feature not class» держится.
- lift (формула 22) + cue + layers → `data/analysis/27c/hac_k12_table8.csv` (lift 6.3-18.8, тот же масштаб).
- acceptance c1-c4,c6 → `data/analysis/27c/hac_k12_acceptance.csv`: **8 accepted\* / 1 partial (C5 hub, abs-cos<0.85) / 3 descriptive (singletons C2,C6,C12)**. (\*pending c5; c4 = proxy group_coverage_frac.)
**Осталось CSD3:** c5 causal (joint ablation IR/SFR — финализирует accepted/partial, у Louvain именно он демотировал C7/C8/C9), ablation-углы (Table 8 angle), C6/L18 fire.

**Апдейт 2026-07-07 (провенанс уточнён):**
- **Канон Table 9 = `data/analysis/27c/hac_k12_table8_renumbered.csv`** (renumbered 1-12 + `agglo_id`-мэппинг на HAC-ID; совпадает с `thesis/main.tex tab:clusters` 1:1). `hac_k12_table8.csv` = не-renumbered (HAC-ID).
- **Cluster-7 (=agglo 6, charge−1, L18) fire/offset=⏳:** L18 нет в activation_matrix (слои L10-17,19-25) → см. стр.71. Единственная строка с пропусками.
- ✅ **Опечатка исправлена:** main.tex Table 9 #6 fire_b `3.47→3.37` (источник hac_k12=3.37).
- Полный провенанс (Table 9 + non-localizability 78/101/114/115 + cluster-7 + hub-5) сведён в `docs/RESULTS_PROVENANCE.md` → секция «P3 §4.3».
- **2 pending до сдачи:** (1) ~~c5 causal~~ ✅ **ГОТОВО (Colab A100, 2026-07-11)** → **6 accepted / 3 partial / 3 descriptive** (было 8/1/3); c5 демотировал #2,#9 (flip <5%). Файлы: `data/results/cluster_joint_ablation/{joint_ablation_physics_decay_type_probe_train.csv, c5_table9_status.csv}`. main.tex Table 9 + Прил.D обновлены. ⚠ c5 = flip-критерий (interaction-плечо не считалось, ~5ч GPU не делали); (2) ~~C6/L18 fire~~ ✅ ЗАКРЫТО + ablation-углы (B.3/B.4) — ещё облако.

**C6/L18 fire ЗАКРЫТО (2026-07-11):** Table 9 #7 (L18_F145795) заполнено **0.49/0.46/0.04** из `decay_cluster_feature_firing.csv` (выход самого пайплайна; cross-check с hac_k12 на L15-singleton = ±0.2, как прочие клетки). Локальный transcoder-encode РАБОТАЕТ (фикс `from __future__ import annotations` в `src/transcoder/activation_functions.py`; layer_N.safetensors в корне repo), но даёт ДРУГУЮ конвенцию, чем CSD3-матрица (per-prompt corr 0.02–0.37, масштаб 3–8× — иной порядок промптов+нормировка) → точные Table-9-числа локально не воспроизводимы, но encode-структура (малые, fire_a>fire_b) подтверждает значение. **Вывод: transcoder-encode локально доступен (fire/attribution), но для ЧИСЛОВОЙ сопоставимости с существующей Table 9 нужен пайплайн-конвент CSD3.**

## 🔬 w_res steering — что есть локально + качественная проверка флипов
**Цель:** вручную убедиться, где steering вдоль w_res РЕАЛЬНО переворачивает ответ, а где intact=0 — артефакт топ-1 (формат), а не инертность. НЕ переписывать §3.1 до прогона.

**Формула (122, проверено):** `margin_flip` смотрит на КОНКРЕТНЫЕ токены классов (`lp[id_β]−lp[id_α]`) — правильная мера; `intact` = топ-1 по словарю ∈ {класс-токены} — формат-зависим (может поймать «The»). `intact_flip = margin_flip И intact`.

**Что ЕСТЬ локально (fallback, если CSD3 не встанет):**
- `data/analysis/runD_v2/<concept>/steering_delta/steering_sweep_tier1.csv` — **АГРЕГАТЫ** (1 строка на layer×c×dir, БЕЗ per-prompt).
- Поля: `margin_flip`, `mean_dmargin_toward` (знаковый сдвиг маржи), `flip_norm`, `flip_norm_intact`, `intact_rate`, `flip_c0_to_c1/c1_to_c0`, `sigma`.
- dir: delta, usage, **w_res**, random0/1, shuffled0/1. c ∈ {1,4,16}.
- Покрытие (10): B1_alpha_beta(scaffold α/β), raw_suffix, B1_grammar_number, particles4_binary/electron_vs_photon(e/γ scaffold), 6 particle_pairs V2.
- **Ключевые агрегаты w_res (c=16, max по слоям):** decay scaffold margin 0.09 / intact 0.07 (rand 0.01) = **инертно чисто**; e/γ scaffold margin 1.0 / intact 0.70 (rand 0.02) = причинно; V2-пары margin 1.0 / **intact 0.00–0.34** (rand-margin 0.27–0.42, грязно); grammar margin 0.94 (rand 0.58, хрупко).
- Пример: decay L24 c16 w_res — `mean_dmargin_toward=0.71` но `margin_flip=0.03` → маржу двигает, но знак почти не переворачивает (инертно).

**Чего НЕТ локально (нужен CSD3):** per-prompt флипы + **реальные сгенерированные токены/ответы** под w_res. Без них нельзя глазами проверить «флип в валидный класс vs «The…»-артефакт».

**Качественный прогон (готовить):** `scripts/steering_decode_check.py` (w_res уже добавлен) — по промптам печатает топ-15 токенов + ранги класс-токенов + 20-токенную генерацию для baseline/delta/usage/**w_res**.
- Прогнать на 3 ключевых: **(1) decay scaffold** (инертно — должно подтвердиться нет-флипа), **(2) e/γ scaffold** (intact 0.70 — проверить, реальный флип или формат знает ответ), **(3) e/γ V2** (margin 1.0/intact 0 — verbose-флип или мусор?).
- Нужны дампы `field_dump_119/` + prompts + steering_csv каждого (на CSD3; пути подтвердить после `git pull`).
- sbatch: `jobs/run_decode_check.sbatch` (расширить на 3 концепта).

## 📝 [ПРОВЕРИТЬ] для тезиса (`~/Downloads/thesis_dillmann_ru-5_edited.md`)
**Закрыто 14/28** локально (значение+источник вписаны). **Остаток 14 — нужен CSD3/нелокальные данные.**
Главная пойманная ошибка: cos(u,γ) было «0.78» = значение K=4-мультикласса, для α/β →0.99 (исправлено).

| строка (≈) | маркер | что нужно / где взять |
|---|---|---|
| 608 | align мультикласс K=4 | дампы K=4 (Прил. C); прогон серии K=4 |
| 612 | декодируемость 99.7% / first-token 8% | K=4 дамп |
| 616 | симплекс 0.41 / 0.04 | K=4 (`scripts/simplex_check.py`) |
| 634 | cos(δ_scaffold, δ_rawsuffix) 0.08→0.70, γ-вычт. 0.66 | дампы scaffold+raw (cross-framing δ) |
| 642 | d′ 2.82 / 1.31 на L18 | decay-probe d-prime источник |
| 496 | сигнал ‖δ‖/‖bulk‖ scaffold/grammar | частицы — локально в `realized_*.csv`; scaffold/grammar — дамп |
| 221 | shuffled-label AUC null | probe-null прогон |
| 737 | j108 совместное направление (AUC/cos) | эксп. j108 |
| 743 | onset-конвергенция 6/10 на L21, полоса L17–24 | Part III onset |
| 777 | проекции δ→used 0.236@L34 / w_res→used ≤0.002 | эксп. j106 |
| 594 | точные пороги критериев кластеров | пайплайн 23/26 |
| 670, 678 | имена скриптов + значения §3.6/§3.7 | 112b / 26 / 27c + данные |
| 630 | разница точности shortcut vs forced | baseline-точность |

**Закрытые (для справки):** cos(w_LDA,u)=0.02, cos(u,γ)→0.99, cos(w_LDA,γ)=0.02, AUC held-out 0.988–0.996 (`axis_causality_summary.json`), d_eff 3–20 (`geometric_atlas.csv`), dual-sign L24_F52031 α+0.63/β−0.89 (`runB`), фичи L21 48u/0w_res (`feat_align`, скрипт 109), cos(δ,u) 0.34/0.02, cos(δ,w_LDA) 0.17/0.27, cov-null переформулирован.

## Линия 1 — 27c: мост Гл.5-кластеры × геометрия (α/β decay)
- **sbatch:** `jobs/run_27c_jobs.sbatch` (array 1–5, n_null=5, want_u, 12ч)
- **Статус задач (2026-06-27):** ✅ task2 (per-feature, `job3_perfeature_geometry.csv`), ✅ task3 (part1), ✅ task5 (part3) — ГОТОВЫ. ⏳ task1 (carriers) + task4 (C4–C7) — таймаут на 5ч, переотправлены на SL2 с лимитом 8ч: `31152827_[1]`, `31152829_[4]`.
- **Назначение задач и выходы (в корне проекта на CSD3):**

| task | что | выход |
|---|---|---|
| 1 | carriers (L24 β / L18 α) | `job1_carriers_geometry.csv` |
| 2 | per-feature (L24,L18) | `job3_perfeature_geometry.csv` |
| 3 | кластеры C0–C3 | `job2_semantic_part1.csv` |
| 4 | кластеры C4–C7 | `job2_semantic_part2.csv` |
| 5 | кластеры C8–C10 | `job2_semantic_part3.csv` |

- **После завершения 3,4,5 — склеить job2:**
```bash
cat job2_semantic_part1.csv > job2_semantic_geometry.csv
tail -n +2 job2_semantic_part2.csv >> job2_semantic_geometry.csv
tail -n +2 job2_semantic_part3.csv >> job2_semantic_geometry.csv
```
- **Урок по времени:** 27c-джоб = ~2.7ч накладных (модель 46мин + baseline 2ч) + группы → лимит ≥8ч обязателен.

---

## Линия 2 — particles 6 пар: δ-вращение + steering (приоритет)
- **sbatch:** `jobs/run_particle_pairs_all.sbatch` (array 1–12)
- **job ID (SL2):** `31138380`…`31138391` (= задачи 1…12)
- **Источник:** срез `data/prompts/particle_pairs/particles_<pair>.jsonl` (из v2-корпуса, 6 пар, сбаланс.)
- **Пары:** 1 e_vs_n, 2 e_vs_γ, 3 e_vs_p, 4 n_vs_γ, 5 n_vs_p, 6 γ_vs_p

| task | линия | выход |
|---|---|---|
| 1–6 | δ-вращение (154→155) | `realized_<pair>.csv` (корень) + дамп `data/analysis/runD_v2/particle_pairs/<pair>/realized_writes/` |
| 7–12 | steering (119→122) | `data/analysis/runD_v2/particle_pairs/<pair>/steering_delta/steering_sweep_tier1.csv` |

- **Статус: ✅ ВСЕ 12 ГОТОВЫ** (2026-06-27). δ-вращение=MLP **82–86% инвариант** на 6 парах (из логов). steering: usage intact→1.0, delta intact≈0 (v2-срез), w_res≈0.
- **⚠ Консистентность:** e/γ в `delta_lever.pdf` — scaffold-срез, тут e/γ — v2-срез → отдельная 6-парная фигура, не подмешивать.
- **Синк CSV (когда ssh пустит):** `bash scripts/sync_pairs_results.sh` (или `SYNC_HOST=iv294@login-q-1... bash …` если fail2ban).

---

## Линия 3 — пайплайн кластеров particles ✅ ГОТОВО (2026-06-27)
**РЕЗУЛЬТАТ:** GPU `04→06→07` (23691 строк абляции) + CPU `09→19→22→23→26` прошли end-to-end на CSD3. Стык 07→19 сработал. **8 latent-state кластеров** (`coimp_louvain`, sil=0.534, 53 фичи) — аналог C0–C10 Гл.5. Выходы на CSD3: `data/results/clustering_particles/cluster_labels.csv` + `cluster_*` + `data/results/grouping/`. **Синкнуть на Mac** (когда ssh пустит). Опционально дальше: 27c-стиль геометрия-абляция на этих 8 кластерах (мост, GPU).

_(ниже — исходный план/цепочка, для справки)_
Цель: particle latent states (аналог Гл.5 C0–C10) для набора **A** (e/n/γ/p).

**ПОЛНАЯ цепочка (проверено по скриптам, зеркало `jobs/run_probe_runB_ablation.sbatch`):**
| стадия | скрипт | где | очередь |
|---|---|---|---|
| 04 | `04_extract_transcoder_features.py` | **GPU** | да |
| 06 | `06_build_attribution_graph.py` (роль-аware граф) | **GPU** | да |
| 07 | `07_run_interventions.py --experiment ablation --per_feature` → `intervention_ablation_*.csv` | **GPU** | да (тяжёлая) |
| 19 | `19_feature_prompt_analysis.py` (пивот abl → `feature_prompt_effect_matrix.csv` и пр.) | **CPU** ✓ | нет |
| 22 | `22_prepare_clustering_inputs.py` | **CPU** ✓ | нет |
| 23 | `23_run_clustering_benchmark.py` (даёт `coimp_louvain`) | **CPU** ✓ | нет |
| 26 | `26_cluster_semantics.py` | **CPU** ✓ | нет |
| 27 | `27_cluster_joint_ablation.py` (валидация) | **GPU** | да (маленькая) |

**Поправлено в резерве (было неверно):** grouping CSV пишет **19** (не 06); GPU-абляция = **07 `--per_feature`** (sbatch её пропускал); 26 НЕ принимает `--behaviour` (только `--grouping_dir/--clustering_dir/--no_dashboard`); header → FERGUSSON/gpu2.

**Решение зафиксировано: вариант (I) co-importance** (как Ч5, через 07-абляцию). co-activation отвергнут — другой критерий, несравнимо с C0–C10.

**Файлы резерва (готовы, синтаксис OK):**
- `jobs/run_particle_clusters_gpu.sbatch` — GPU 04→06→07.
- `scripts/run_particle_clusters_local.sh` — CPU 19→22→23→26.

**⚠ Подтвердить ПЕРЕД отправкой (2 behaviour-специфичных параметра):**
1. **Путь abl-CSV от 07** → в локальном раннере `ABL_CSV=` (07 пишет в `data/results/interventions/particles/...`; проверить точное имя после прогона 06/07).
2. **`--require_n_graph_features`** в 07 — у decay было 69; для particles неизвестно до 06 → пока опущено (если 07 ругнётся, выставить из графа).

**Порядок:**
1. ✅ Промпты — генерировать НЕ нужно (корпус `physics_internal_candidate_selection_v2` уже есть, 487).
2. ✅ GPU sbatch + локальный раннер написаны.
3. ⬜ Отправить `run_particle_clusters_gpu.sbatch` — **ПОСЛЕДНИМ**, когда очередь (17 джобов) разгрузится.
4. ⬜ Синк `data/results/grouping/` + граф → `bash scripts/run_particle_clusters_local.sh` локально.
5. ⬜ Построить `cluster_membership_particles.csv` → 27c-геометрия на particle-кластерах (carrier_geom_core).

**Статус:** резерв готов, держим. Слать GPU-часть последней. Если за 5 дней не успеет — не страшно (усиление, не закрытие дыры; трихотомия на particles уже работает через δ-вращение+steering).

## Как забрать результаты (синк с CSD3 на Mac)
```bash
H=iv294@login.hpc.cam.ac.uk; R=/rds/user/iv294/hpc-work/thesis/project
# 27c
rsync -avz "$H:$R/job1_carriers_geometry.csv" "$H:$R/job3_perfeature_geometry.csv" "$H:$R/job2_semantic_geometry.csv" data/analysis/27c/
# particles δ-вращение
rsync -avz "$H:$R/realized_*.csv" data/analysis/runD_v2/particle_pairs/
# particles steering
rsync -avz "$H:$R/data/analysis/runD_v2/particle_pairs/" data/analysis/runD_v2/particle_pairs/
```

## Что делаю по приходу результатов
- **27c:** склеить part1–3 → свести с `orient_delta` кластеров C0–C10 → Results «latent states × geometry» + MASTER §4.13.
- **δ-вращение 6 пар:** свести MLP% × пара (варьирует ли с парой) → MASTER §4.12.
- **steering 6 пар:** проверить иерархию u<δ<w_res на всех парах (особенно n/p — труднейшая) → 6-парная фигура.

## Очередь / контроль
```bash
squeue -u iv294        # R=бежит, PD=ждёт GPU
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode   # итог по конкретному
```
QOS `gpu2` ограничивает число одновременных GPU → часть стоит PD, стартуют по мере освобождения. Все джобы независимы и укладываются в свои лимиты.

## 🔬 Rotation ↔ Causality — методы + почему пока INCONCLUSIVE (сохранено 2026-07-01)
Проверяли: крутёж/длина осей связаны ли с причинностью по слоям? Скрипты: `rotation_causality_link.py` (v1), `rotation_causality_v2.py` (angle vs length), `rotation_null_check.py`, `norm_causality.py`. **Пока УБРАНО из Appendix C** (слабо, не уверены в экспериментах). Если вернёмся — сделать правильные тесты (raw steering ∝‖u‖ / интервенция на норму).

### Три теста (rotation_causality_link.py) — методы
**Test 1 — Onset ordering (крутёж раньше причинности?):**
1. Две кривые по слоям: причинность(ℓ) = delta-флип от steering (c=16); крутёж(ℓ) = поворот δ на слое.
2. L_causal = первый слой где флип ≥ 0.5.
3. frac_before = Σ(крутёж для ℓ<L_causal) / Σ(крутёж все ℓ).
4. Высокая → крутёж «докрутился, потом причинен»; ~половина → одновременно; низкая → причинность раньше.

**Test 2 — Detrended correlation (убрать «оба растут с глубиной»):**
1. Проблема: крутёж и флип оба растут с глубиной → тривиально коррелируют.
2. Для каждой кривой: линейная регрессия vs номер слоя, вычесть → остатки (не объяснённые глубиной).
3. Spearman на остатках.
4. Выживает → реальная связь сверх тренда; исчезает→0 → тривиальная.

**Test 3 — Lagged cross-correlation на скоростях:**
1. Если крутёж ведёт → скорость крутёжа(ℓ) предсказывает скорость флипа(ℓ+lag), lag>0.
2. Берём приращения (rates) — пиковые, честнее монотонных.
3. Перебор lag, максимизируем корреляцию; калибровка против shuffled-layer null.
4. lag>0 → крутёж ведёт.

### Реальные числа (scaffold, пересчёт 2026-07-01) — почему INCONCLUSIVE
- L_causal (delta-flip≥0.5) = **L22**.
- **Test1 frac_before: norm-free ANGLE = 81% vs rot_mag(норм-взвеш, БАГ) = 6%** — метрика МЕНЯЕТ ответ радикально.
- **Test2 Spearman: raw −0.74 → detrended −0.49** — ОТРИЦАТЕЛЬНАЯ (крутёж рано, флип поздно, антикоррелируют, не совпадают по слоям).
- **Test3 best lag=2, corr=0.13** — почти ноль, нет «крутёж ведёт».
- ⚠ Моё 81% ≠ мемо 29-56% → **точная метрика крутёжа сильно влияет** → само по себе доказывает fragility.

### norm→causality — почему НЕ протестировано (важно)
Steering нормирован: h ← h + c·σ·û (единичное û, магнитуда c·σ **независимо от ‖u‖**). Значит «cut ‖u‖ и steer» бессмысленно (û_cut=û, σ сокращает масштаб). Флип vs ‖u‖ (flip L17-23, ‖u‖ half-max L12-14) НЕ доказывает «слой не норма» — норму нормированным steering не проверить. Нужен **raw steering (магнитуда ∝‖u‖)** или интервенция на ‖u‖ — НЕ делалось.

**ИТОГ: НЕ «нет связи» и НЕ «есть связь» — INCONCLUSIVE** (метрико-зависимо, контроли слабые, норму не тестировали правильно). Трихотомия стоит на норме/механизме/тайминге (не на rotation) — это отдельно и надёжно.
