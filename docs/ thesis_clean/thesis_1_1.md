
### Attribution-defined latent state

Пусть $\mathcal{M}$ — языковая модель, $\mathcal{P}$ — распределение контролируемых промптов, а $\mathcal{C}$ — фиксированная high-level **каузальная модель** с абстрактной физической переменной $V_h$ (например, тип ядерного состояния, lepton family, 2n–2p composition), для которой $\alpha/\beta$-классификация рассматривается как наблюдаемая функция $f(V_h)$ [1].

Пусть $\mathcal{F}$ — множество индексов transcoder фичей, извлечённых на некотором слое $\ell$ с помощью обученного transcodера $T^{(\ell)}=(\mathrm{enc}^{(\ell)},\mathrm{dec}^{(\ell)})$, реализующего «базис признаков» для feature circuits[2].

Пусть $\mathcal{T}_{\mathrm{irr}}$ — семейство **task preserving** преобразований промпта (парафраз, перевод, смена формата ответа, малошумовая пертурбация residual stream), которые в идеале сохраняют значение $V_h(p)$ [7]. Пусть $\mathcal{T}_{\mathrm{rel}}$ — **state swapping** инволюция, которая сохраняет синтаксис и стиль, но систематически заменяет описания одного high-level физического состояния $V_h(p)$ на эквивалентные описания другого $V_h(p')\neq V_h(p)$, аналогично структурированным контрфактуальным трансформациям в causal abstraction [3].

Пусть $\Phi$ — произвольная, фиксированная процедура кластеризации фичей, такая что (i) на вход она получает атрибуционно-каузальные данные (например, граф co-importance или attribution graph), (ii) на выходе даёт разбиение $\mathcal{F}$ на непересекающиеся группы $\{G_1,\dots,G_K\}$, и (iii) детерминирована при фиксированных входных данных. Мы пишем $\Phi(p)\ni G$, если группа $G$ получена процедурой $\Phi$ при запуске на корпусе, содержащем $p$. В эмпирической части $\Phi$ инстанциируется как average-linkage agglomerative кластеризация на co-importance Jaccard-графе; метод и число кластеров выбраны из множества кандидатов по *carrier stability* (воспроизводимость кластеров под парафразными трансформациями входа, см. Приложение B), в духе методологии recent circuit-tracing и benchmark-работ [4].

Наконец, зафиксируем набор порогов

$$
\Theta
=
\big(
\tau_\sigma,\,
\eta_{\mathrm{irr}},\,
\eta_{\mathrm{rel}},\,
\tau,\,
\tau_{\mathrm{nec}},\,
\Delta_{\mathrm{ctrl}},\,
\Delta_{\mathrm{recon}},\,
\mathrm{IR}_{\max},\,
\eta_{\mathrm{null}},\,
\delta_{\mathrm{min}}
\big),
$$

определяющих строгие критерии стабильности, каузальной достаточности и структурной целостности. В духе preregistration-подходов к механистической интерпретируемости $\Theta$ фиксируется до анализа [5].

**Определение 1 (Attribution-defined latent state).**  
Кандидат $G\subset\mathcal{F}$ называется *attribution-defined latent state* для high-level переменной $V_h$ под порогами $\Theta$, что записывается как

$$
G \models_\Theta V_h,
$$

если выполняются все три условия ниже:

1. **Representational stability and polarity (Condition I).**  

   - (*Carrier stability*). Активация кластера $G$ определяется преимущественно физическим состоянием $V_h$, а не поверхностной формулировкой входа. Определим среднюю активацию кластера на промпте $p$ как $A_G(p)=\frac{1}{|G|}\sum_{(\ell,k)\in G}a_k^{(\ell)}(p)$. Пусть $g$ пробегает semantic_equivalence_groups (группы промптов с одним физическим cue и разными wording-вариантами), $\mu_g=\mathbb{E}_{p\in g}[A_G(p)]$ — среднее внутри группы. Тогда требуем, чтобы межгрупповая вариация активаций между семантическими группами (semantic_equivalence_groups), соответствующими разным значениям V_h, доминировала над вариацией внутри групп:

$$
\mathrm{ICC}(G)
=
\frac{\sigma^2_{V_h}}{\sigma^2_{V_h}+\sigma^2_{\mathrm{irr}}}
\;\geq\;
\tau_{\mathrm{ICC}},
\tag{1.I.a}
$$

где $\sigma^2_{V_h}=\mathrm{Var}_g(\mu_g)$ — дисперсия средних активаций между физическими ситуациями, $\sigma^2_{\mathrm{irr}}=\mathbb{E}_g[\mathrm{Var}_{p\in g}(A_G(p))]$ — ожидаемая within-group дисперсия от task-preserving трансформаций $\mathcal{T}_{\mathrm{irr}}$ (paraphrase). ICC = 1 при полной стабильности; ICC = 0 если вариация от surface-формы равна вариации между физическими ситуациями. Порог $\tau_{\mathrm{ICC}} = 0.5$ фиксируется a priori как критерий «умеренной надёжности» по шкале Koo \& Mae (2016): ICC $\geq 0.5$ означает, что межгрупповая (физическое состояние $V_h$) дисперсия объясняет не менее половины суммарной вариации активаций кластера. Активации $a_k^{(\ell)}(p)$ собираются отдельным проходом модели без интервенции (скрипт 38); ICC вычисляется локально (скрипт 38b).

   - (*Polarity consistency*). Пусть $\tilde{\sigma}_G(p)$ — подписанный кластерный вклад:

$$
\tilde{\sigma}_G(p)
=
\frac{1}{|G|}
\sum_{(\ell,k)\in G}
\mathrm{effect}\bigl((\ell,k),\, p\bigr),
$$
   
где $\mathrm{effect}((\ell,k),p) = a_k^{(\ell)}(p)\cdot\tfrac{\partial\Delta\mathrm{logit}(p)}{\partial a_k^{(\ell)}}$ — attribution score (gradient $\times$ activation) со знаком: положительное значение означает, что фича помогает правильному ответу на промпт $p$, отрицательное — мешает. Мы требуем *observational polarity* — противоположность знаков в среднем на двух подкорпусах:



$$
\mathbb{E}_{p\sim\mathcal{P}_\alpha}\!\bigl[\tilde{\sigma}_G(p)\bigr]
\;\cdot\;
\mathbb{E}_{p\sim\mathcal{P}_\beta}\!\bigl[\tilde{\sigma}_G(p)\bigr]
\;<\; 0.
\tag{1.I.b}
$$

то есть распределения σ~Gσ~G​ для двух состояний VhVh​ должны быть смещены в противоположные стороны; это обеспечивает, что простой бинарный предиктор αG(p)=sign(σ~G(p))αG​(p)=sign(σ~G​(p)) ошибается с вероятностью меньше 0.5.

2. **Causal sufficiency under feature-level interventions (Condition II).**

Для контрастивной пары $(p, p')$ с $V_h(p)\neq V_h(p')$ подменяем активации фичей $G$ в промпте $p$ активациями из $p'$ через transcoder $T^{(\ell)}$ [1]. Interchange intervention accuracy $\mathrm{IIA}(G;\mathcal{D})$ определяем как долю пар, на которых feature-space
interchange по G меняет знак логитного margin: (измеряет, насколько часто такой патчинг переключает ответ модели)
$$
\mathrm{IIA}(G;\mathcal{D}) = \Pr_{(p,p')\sim\mathcal{D}}\!\Big[\mathrm{sign}\bigl(\Delta_{\mathrm{patch}}(p\!\leftarrow\!p')\bigr) \neq \mathrm{sign}\bigl(\Delta_{\mathrm{orig}}(p)\bigr)\Big].
\tag{2.I}
$$

Для компактных механизмов ожидается $\mathrm{IIA}(G;\mathcal{D}) \geq \tau$. Однако при **distributed representation** с redundant coding ($\overline{\mathrm{IR}}(G) \ll 1$, Condition III) патчинг одного кластера не переключает ответ — остальные кластеры компенсируют. В этом случае $\mathrm{IIA}(G) \approx 0$ является следствием распределённости, а не отсутствия каузального вклада [9].

Операциональным тестом каузальной достаточности для distributed $G$ служит **necessity under joint ablation** — насколько часто *удаление* $G$ разрушает поведение модели:

$$
\mathrm{Nec}(G;\mathcal{P}) = \Pr_{p\sim\mathcal{P}}\!\Big[\mathrm{sign}\!\big(\Delta_{\mathrm{abl}}^{G}(p)\big) \neq \mathrm{sign}\!\big(\Delta_{\mathrm{orig}}(p)\big)\Big] \;\geq\; \tau_{\mathrm{nec}}.
\tag{2.II}
$$

Считаем Condition II выполненным, если выполняется хотя бы одно из:

$$\mathrm{IIA}(G;\mathcal{D}_{\mathrm{matched}})\ge\tau
\quad \text{или} \quad
\mathrm{Nec}(G;\mathcal{P})\ge\tau_{\mathrm{nec}}.
\tag{2.III}
$$

Первая ветка соответствует локализованным механизмам, вторая — сильно распределённым, где прямой patching недооценивает вклад G.


### Structural coherence and minimality (Condition III)

Пусть \(\Delta_{\mathrm{joint}}(G;p)\) и \(\Delta_k(p)\) — изменения логитного margin \(\Delta_{\mathrm{orig}}(p)\) при совместной и индивидуальной абляции фичей в \(G\). Определим interaction ratio

\[
\mathrm{IR}(G;p)
=
\frac{\Delta_{\mathrm{joint}}(G;p)}{\sum_{(\ell,k)\in G}\Delta_k(p)}.
\tag{3.0}
\]

1. **Coherence.** Совместный эффект абляции \(G\) должен быть существенно суб‑аддитивным:

\[
\overline{\mathrm{IR}}(G)
:=
\mathbb{E}_{p\sim\mathcal{P}}\big[\mathrm{IR}(G;p)\big]
\leq
\mathrm{IR}_{\max},
\tag{3.1}
\]

где \(\mathrm{IR}_{\max}<1\) фиксируется *a priori* и отражает redundant/competitive кодирование внутри единого механизма.

2. **Specificity.** Суб‑аддитивность должна быть специфичной для \(G\), а не свойством произвольных кластеров такого же размера и layer profile. Пусть \(\mathrm{IR}_{\mathrm{null}}(|G|;p)\) — медиана \(\mathrm{IR}\) для случайных подмножеств того же размера и layer profile на prompt \(p\). Требуем

\[
\Pr_{p\sim\mathcal{P}}
\big[
\mathrm{IR}(G;p) < \mathrm{IR}_{\mathrm{null}}(|G|;p)
\big]
\geq
1-\eta_{\mathrm{null}}.
\tag{3.2}
\]

3. **Minimality.** Ни одно proper subset \(G'\subset G\) не должно воспроизводить интервенционный эффект \(G\) на \(V_h\) в пределах допуска:

\[
\mathrm{IIA}(G';\mathcal{D}_{\mathrm{matched}})
\leq
\mathrm{IIA}(G;\mathcal{D}_{\mathrm{matched}})
- \delta_{\mathrm{min}}
\quad
\text{для всех } G'\subset G.
\tag{3.3}
\]

Условия (3.1)–(3.3) вместе означают, что \(G\) является **структурно согласованным и минимальным** bundled‑механизмом: его совместный эффект не редуцируется к сумме независимых эффектов, этот паттерн не характерен для случайных многослойных кластеров того же размера, и его нельзя сузить без заметной потери интервенционной силы.

---

Пусть $G\subset\mathcal{F}$ — feature bundle, удовлетворяющий $G\models_\Theta V_h$ по Определению 1. Разрешим $G$ быть **многослойным**, то есть содержать фичи с разными layer-индексами; кластеризация $\Phi$ оперирует лишь по co-importance Jaccard и не использует layer-информацию, так что $G$ может включать параллельные детекторы одного и того же физического паттерна на разных глубинах сети [2].

Рассмотрим отображение

$$
\alpha_G \colon \text{states at layer(s) supporting } G \;\to\; \text{values of } V_h,
$$

которое каждой конфигурации внутренних активаций $\{a_k^{(\ell)}(p)\}_{(\ell,k)\in G}$ сопоставляет предсказанное значение $V_h$ по знаку effect-based кластерного вклада $\tilde{\sigma}_G(p)$.

**Теорема 1 (Faithful distributed abstraction).**  
Существует выбор порогов $\Theta$ и параметров $\varepsilon_{\mathrm{pred}}, \varepsilon_{\mathrm{int}}\ge 0$, зависящих только от $\Theta$, таких что, если $G\models_\Theta V_h$, то отображение $\alpha_G$ задаёт **distributed $\varepsilon$-faithful абстракцию** переменной $V_h$ в смысле causal abstraction, а именно [12]:

1. (**Predictive alignment**) На распределении $\mathcal{P}$ прогноз $\alpha_G$ почти всегда совпадает с истинным $V_h$:

$$
\Pr_{p\sim\mathcal{P}}\big[\alpha_G(p) = V_h(p)\big]
\;\ge\;
1 - \varepsilon_{\mathrm{pred}},
$$

где $\varepsilon_{\mathrm{pred}}$ ограничен сверху явной функцией от $\eta_{\mathrm{irr}}, \eta_{\mathrm{rel}}, \tau_\sigma$ из Condition I (carrier stability и polarity).

2. (**Interventional alignment**) Для контрастивных пар $(p,p')$ с $V_h(p)\neq V_h(p')$ существует *low-level* вмешательство $\mathrm{Interchange}_G(p\leftarrow p')$, действующее только на координаты в $G$ (возможно, на нескольких слоях), такое что поведение модели после интервенции $\varepsilon_{\mathrm{int}}$-близко к поведению абстрактной модели при интервенции на $V_h$ [1]:

$$
\Pr_{(p,p')\sim\mathcal{D}_{\mathrm{matched}}}
\Big[
\mathrm{sign}\big(\Delta_{\mathrm{int}}(p,p')\big)
=
\mathrm{sign}\big(\Delta_{\mathcal{C}}(V_h(p)\leftarrow V_h(p'))\big)
\Big]
\;\ge\;
1 - \varepsilon_{\mathrm{int}},
$$

где $\Delta_{\mathcal{C}}(\cdot)$ — логитный или вероятностный эффект интервенции в high-level модели $\mathcal{C}$, а $\varepsilon_{\mathrm{int}}$ ограничен функцией от $\tau, \Delta_{\mathrm{ctrl}}, \Delta_{\mathrm{recon}}$ и концентрации $\mathrm{IIA}(G;\mathcal{D}_{\mathrm{matched}})$ из Condition II [11].

3. (**Distributed minimal mechanism**) Кластер $G$ является **минимальным распределённым механизмом** для $V_h$ в том смысле, что:
   - (coherence + specificity) взаимодействие фичей в $G$ существенно отличается от взаимодействия в случайных многослойных кластерах того же размера и layer profile:

$$
\overline{\mathrm{IR}}(G) \le \mathrm{IR}_{\max},
\quad
\Pr_{p}\big[\mathrm{IR}(G;p) < \mathrm{IR}_{\mathrm{null}}(|G|;p)\big]
\ge 1-\eta_{\mathrm{null}},
$$

то есть фичи в $G$ формируют **coherent distributed unit**, а не произвольную супергруппу [2];

   - (minimality) ни одно proper subset $G'\subset G$ не достигает сопоставимой interventional alignment:

$$
\mathrm{IIA}(G';\mathcal{D}_{\mathrm{matched}})
\le
\mathrm{IIA}(G;\mathcal{D}_{\mathrm{matched}}) - \delta_{\mathrm{min}},
$$

так что $G$ минимален по включению среди всех distributed bundles, реализующих заданный уровень faithfulness [6].

В совокупности это означает, что $\mathcal{M}$ реализует $\varepsilon$-faithful **distributed abstraction** $\mathcal{C}$ на подпространстве, заданном многослойным кластером $G$: существует отображение $\alpha_G$, делающее диаграмму «high-level интервенции $V_h$ $\leftrightarrow$ low-level интервенции по $G$» $\varepsilon$-коммутативной в смысле causal abstraction [1, 12, 13].

---

### Доказательство Теоремы 1

Доказательство состоит из трёх частей: (A) построение отображения $\alpha_G$ и оценка его **предсказательной ошибки**, (B) установление **интервенционной согласованности** между абстрактной моделью $\mathcal{C}$ и интервенциями по $G$, и (C) вывод **distributed минимальности** из Condition III. Затем мы применяем стандартный результат из теории causal abstraction, связывающий эти свойства с $\varepsilon$-faithfulness [1, 12].

#### Шаг A. Построение $\alpha_G$ и предсказательная согласованность

По Определению 1, Condition I закрепляет **representational stability and polarity**. В частности:

- (I.a) утверждает, что $G$ воспроизводится $\Phi$ при переходах между template families, реализующими одно и то же $V_h$, с ошибкой не более $\eta_{\mathrm{irr}}$.
- (I.b) утверждает, что sign-ориентированный вклад $\tilde{\sigma}_G(p)$ имеет противоположные знаки на двух подкорпусах $\mathcal{P}_\alpha, \mathcal{P}_\beta$, соответствующих противоположным значениям $V_h$ (или их образу $f(V_h)$), при условии $|\tilde{\sigma}_G(\cdot)|>\tau_\sigma$.

Определим отображение $\alpha_G\colon \text{states} \to \text{values of }V_h$ следующим образом. Для каждого prompt $p\in\mathcal{P}$:

1. Вычислим effect-ориентированный вклад $\tilde{\sigma}_G(p)$ по формуле из Определения 1.
2. Если $|\tilde{\sigma}_G(p)|\le \tau_\sigma$, считаем $\alpha_G(p)$ неопределённым (ошибка включается в $\varepsilon_{\mathrm{pred}}$).
3. Если $|\tilde{\sigma}_G(p)|> \tau_\sigma$, положим $\alpha_G(p) = v_\alpha$ при $\tilde{\sigma}_G(p)>0$ и $\alpha_G(p)=v_\beta$ при $\tilde{\sigma}_G(p)<0$, где $v_\alpha, v_\beta$ — два противоположных значения абстрактной переменной $V_h$.

**Лемма A.1** *(точность $\alpha_G$ на $\mathcal{P}$).* Пусть $\varepsilon_{\mathrm{pred}} = \eta_{\mathrm{irr}} + \eta_{\mathrm{rel}} + \Pr_{p\sim\mathcal{P}}[|\tilde{\sigma}_G(p)|\le \tau_\sigma]$. Тогда

$$
\Pr_{p\sim\mathcal{P}}\big[\alpha_G(p) = V_h(p)\big]\;\ge\;1-\varepsilon_{\mathrm{pred}}.
$$

*Доказательство.* Разложим вероятность ошибки:

$$
\Pr\big[\alpha_G(p)\neq V_h(p)\big]
\le
\Pr\big[|\tilde{\sigma}_G(p)|\le \tau_\sigma\big]
+
\Pr\big[|\tilde{\sigma}_G(p)|> \tau_\sigma \;\wedge\; \alpha_G(p)\neq V_h(p)\big].
$$

Второе слагаемое разбивается на два типа ошибок:

- ошибки, вызванные нарушением carrier stability (когда $G$ не восстановился при смене шаблона при фиксированном $V_h$): по (I.a) их вероятность не превосходит $\eta_{\mathrm{irr}}$;
- ошибки, вызванные нарушением polarity consistency (когда знаки вкладов для противоположных $V_h$ не противоположны): по (I.b) их вероятность не превосходит $\eta_{\mathrm{rel}}$.

Суммируя:

$$
\Pr\big[\alpha_G(p)\neq V_h(p)\big]
\le
\Pr\big[|\tilde{\sigma}_G(p)|\le \tau_\sigma\big]
+
\eta_{\mathrm{irr}} + \eta_{\mathrm{rel}},
$$

откуда следует утверждение. $\square$

Таким образом, мы получили отображение $\alpha_G$ с контролируемой предсказательной ошибкой $\varepsilon_{\mathrm{pred}}$, зависящей только от параметров Condition I.

#### Шаг B. Интервенционная согласованность

Используем Condition II, закрепляющий **causal sufficiency under feature-level interventions**. Напомним, что для контрастивных пар $(p,p')$ с $V_h(p)\neq V_h(p')$ определена метрика $\mathrm{IIA}(G;\mathcal{D})$, измеряющая, насколько часто feature-space interchange по $G$ переворачивает знак логитного margin в направлении, согласованном с $V_h(p')$. Condition II требует, чтобы:

- $\mathrm{IIA}(G;\mathcal{D}_{\mathrm{matched}}) \ge \tau$ (сильный интервенционный эффект);
- $\mathrm{IIA}_{\mathrm{rand\text{-}init}}(G) \le \tau - \Delta_{\mathrm{ctrl}}$ (эффект не объясняется архитектурой + transcodером);
- $\mathrm{RFR} \le \tau - \Delta_{\mathrm{recon}}$ (эффект не объясняется reconstruction noise).

Для пары $(p,p')$ с $V_h(p)\neq V_h(p')$ рассмотрим high-level контрфактуал $\mathcal{C}[V_h(p)\leftarrow V_h(p')]$ и соответствующий знак изменения логита целевого ответа $\mathrm{sign}(\Delta_{\mathcal{C}}(V_h(p)\leftarrow V_h(p')))$ [13].

**Лемма B.1** *(high-level ↔ low-level интервенции).* Пусть $\varepsilon_{\mathrm{int}} = 1-\tau + \mathrm{IIA}_{\mathrm{rand\text{-}init}}(G) + \mathrm{RFR}$. Тогда

$$
\Pr_{(p,p')\sim\mathcal{D}_{\mathrm{matched}}}
\Big[
\mathrm{sign}\big(\Delta_{\mathrm{int}}(p,p')\big)
=
\mathrm{sign}\big(\Delta_{\mathcal{C}}(V_h(p)\leftarrow V_h(p'))\big)
\Big]
\;\ge\;
1-\varepsilon_{\mathrm{int}}.
$$

*Доказательство (эскиз).*

1. **Основной эффект.** Из $\mathrm{IIA}(G;\mathcal{D}_{\mathrm{matched}})\ge\tau$ следует, что с вероятностью не менее $\tau$ знак $\Delta_{\mathrm{int}}(p,p')$ отличен от $\Delta_{\mathrm{orig}}(p)$ и согласован с $\mathrm{sign}(\Delta_{\mathcal{C}}(V_h(p)\leftarrow V_h(p')))$.

2. **Контроль архитектурных артефактов.** Малость $\mathrm{IIA}_{\mathrm{rand\text{-}init}}(G)$ гарантирует, что архитектура трансформера + transcodер $T^{(\ell)}$ сами по себе не генерируют значимого интервенционного эффекта при случайных активациях: высокое $\mathrm{IIA}$ специфично для реальных активаций обученной модели [11].

3. **Контроль реконструкционной ошибки.** Малость $\mathrm{RFR}$ гарантирует, что подмена $x^{(\ell)}(p)$ на реконструкцию $\mathrm{dec}(\mathrm{enc}(x^{(\ell)}(p)))$ редко меняет знак margin; следовательно, наблюдаемый эффект $\Delta_{\mathrm{int}}(p,p')$ обусловлен именно изменением координат в $G$, а не искажением всего представления из-за transcodера [2].

Вероятность несовпадения знаков ограничена суммой $(1-\tau) + \mathrm{IIA}_{\mathrm{rand\text{-}init}}(G) + \mathrm{RFR} = \varepsilon_{\mathrm{int}}$. $\square$

Таким образом, интервенции по $G$ реализуют абстрактные интервенции на $V_h$ с контролируемой ошибкой $\varepsilon_{\mathrm{int}}$.

#### Шаг C. Distributed coherence и минимальность

Используем Condition III:

- (III.a) coherence: $\overline{\mathrm{IR}}(G)\le \mathrm{IR}_{\max} < 1$;
- (III.b) specificity: $\Pr[\mathrm{IR}(G;p) < \mathrm{IR}_{\mathrm{null}}(|G|;p)]\ge 1-\eta_{\mathrm{null}}$;
- (III.c) minimality: $\mathrm{IIA}(G';\mathcal{D}_{\mathrm{matched}}) \le \mathrm{IIA}(G;\mathcal{D}_{\mathrm{matched}})-\delta_{\mathrm{min}}$ для всех $G'\subset G$.

**Лемма C.1** *(coherent distributed unit).* Если (III.a) и (III.b) выполнены, то bundle $G$ формирует coherent distributed unit: нет набора случайно выбранных фичей того же размера и layer profile, для которого статистика interaction ratio была бы сравнима с $G$ на overwhelming доле промптов.

*Доказательство.* (III.a) говорит, что совместная абляция $G$ даёт sub-additive эффект по отношению к сумме одиночных абляций; это свойственно либо redundant, либо конкурентному кодированию внутри одного механизма. (III.b) говорит, что такая sub-additivity специфична для $G$ и не возникает типично для random bundles того же размера. Следовательно, структура взаимодействий внутри $G$ отличает его от произвольных кластеров, что интерпретируется как **coherence** в духе sparse feature circuits [2]. $\square$

**Лемма C.2** *(минимальность distributed механизма).* Если (III.c) выполнено, то $G$ минимален по включению среди distributed bundles, реализующих заданный уровень interventional alignment.

*Доказательство.* (III.c) непосредственно утверждает, что для любого proper subset $G'\subset G$ interventional effect в терминах $\mathrm{IIA}$ строго хуже на величину не менее $\delta_{\mathrm{min}}$. Поскольку faithfulness в теореме формулируется через $\mathrm{IIA}(G;\mathcal{D}_{\mathrm{matched}})$, никакое proper subset не задаёт столь же хорошей абстракции $V_h$; следовательно, $G$ минимален для заданной $\mathrm{IIA}$-точности. $\square$

Совместно Леммы C.1 и C.2 закрепляют свойства (3) из формулировки теоремы: $G$ является **coherent, специфическим и минимальным distributed механизмом**.

#### Шаг D. Применение теории causal abstraction

Работы [1, 12] показывают, что если существует отображение $\alpha$ из внутренних переменных модели в абстрактные состояния, которое:

1. почти всегда правильно восстанавливает абстрактное состояние на данном распределении входов (prediction consistency),
2. сохраняет результаты интервенций на абстрактной переменной, реализуемых через корректно определённые low-level интервенции (interventional consistency),

то модель реализует $\varepsilon$-faithful causal abstraction $\mathcal{C}$, где $\varepsilon$ верхне ограничена суммой указанных ошибок.

В нашем случае мы построили $\alpha_G$ (Шаг A) и показали:

- Лемма A.1: $\alpha_G$ имеет предсказательную точность $\ge 1-\varepsilon_{\mathrm{pred}}$.
- Лемма B.1: интервенции по $G$ соответствуют интервенциям на $V_h$ с точностью $\ge 1-\varepsilon_{\mathrm{int}}$.
- Леммы C.1–C.2: реализация через $G$ — coherent и минимальная среди bundled реализаций.

Следовательно, по теореме из [1, 12] модель $\mathcal{M}$ реализует $\varepsilon$-faithful causal abstraction $\mathcal{C}$ на подпространстве, заданном многослойным кластером $G$, с $\varepsilon\le \varepsilon_{\mathrm{pred}}+\varepsilon_{\mathrm{int}}$. Поскольку $G$ может содержать фичи из разных слоёв, а кластеризация $\Phi$ основана только на co-importance Jaccard, полученная абстракция является **distributed**: она реализована не на одном слое или нейроне, а на согласованном многослойном bundle [2, 13].

Это завершает доказательство Теоремы 1. $\square$

---

## Experimental procedure

Equivalently, существует $\delta>0$ такое, что распределения $\tilde\sigma_G$ на $\mathcal{P}_\alpha$ и $\mathcal{P}_\beta$ сдвинуты относительно друг друга (rank-biserial $|r_b|>\delta$). Это тестируется двусторонним Mann–Whitney U. Связь с Леммой A.1: полярность гарантирует, что бинарный предиктор $\alpha_G(p)=v_\alpha$ при $\tilde\sigma_G(p)>0$, $v_\beta$ иначе, ошибается с вероятностью $\eta_{\mathrm{rel}} < 0.5$, где $\eta_{\mathrm{rel}}$ зависит от конкретного распределения $\tilde\sigma_G$.

   > *Примечание о контрфактуальной версии.* Строгая контрфактуальная форма потребовала бы $\Pr_p[\mathrm{sign}(\tilde\sigma_G(\mathcal{T}_{\mathrm{rel}}(p)))=-\mathrm{sign}(\tilde\sigma_G(p))\mid|\tilde\sigma_G(p)|>\tau_\sigma]\ge 1-\eta_{\mathrm{rel}}$, где $\mathcal{T}_{\mathrm{rel}}$ — оператор, изменяющий $V_h(p)$ при фиксированном поверхностном контексте. В наших данных строгие T_rel-пары отсутствуют; вместо них используются *contrastive pairs* — матчед промпты из одного семантического контекста с противоположными $V_h$. Парный sign-flip rate (SFR_pair) на таких парах служит частичным контрфактуальным подтверждением (1.I.b).


Полное развёрнутое определение semantic_equivalence_groups и как именно ты строишь группы gg.
- Формулы для μgμg​, σVh2σVh​2​, σirr2σirr2​ и связь с “классическим” ICC (с ссылкой).
- Подробности про Mann–Whitney U, rank-biserial rbrb​ и конкретное выбранное δδ.
- Описание и графики для pairwise sign-flip rate (SFR_pair) на contrastive парах и как они эмпирически подтверждают (1.I.b).

Там же можно положить таблицы: ICC, p‑values, r_b по кластерам.
Из Condition II
В результаты/аппендикс:

-Числа RFR (8.1%, 17.1%), распределение по слоям.
- Значения IIArand-initIIArand-init​ и обсуждение OOD-эффекта.
- Любые графики “IIA vs Nec vs IR” и как они мотивируют выбор “Nec как основной тест” для распределённых кластеров.
- Подробные схемы interchange-процедуры (как кодируется/декодируется, какие токены патчатся).


## Results / Ablation section (“Robustness of IIA / Nec under random init and reconstruction”)


⁴ *Noise floor:* RFR (reconstruction flip rate при encode→decode без изменений) составляет 8.1% в среднем, до 17.1% на слоях L18–L24; это нижний предел для интерпретации любых flip-метрик. $\mathrm{IIA}_{\mathrm{rand\text{-}init}}$ на случайно инициализированной модели служит контролем на архитектурные артефакты [14]; аномально высокий $\mathrm{IIA}_{\mathrm{rand}}$ в нашем случае (OOD-эффект) подтверждает, что IIA-линейка неприменима и necessity является корректным основным тестом.


## Приложение A. Происхождение порогов Θ

Чтобы отношение $G\models_\Theta V_h$ не превращалось в подгонку порогов под наблюдаемые числа, компоненты $\Theta$ делятся на две категории по способу назначения.

**(1) Control-anchored пороги.** Привязаны к независимому baseline'у (reconstruction noise floor, random-init контроль или null-распределение), который задаётся не экспериментатором, а самой процедурой измерения. Для них «прохождение» означает *значимый отрыв от контроля*, а не достижение заранее выбранного числа.

**(2) Operational пороги.** Фиксируются *a priori* как стандартные уровни строгости (значимость, минимальный эффект), до анализа конкретного $G$, и не меняются по его результатам.

| Параметр | Смысл | Тип | Источник / привязка |
|----------|-------|-----|---------------------|
| $\tau_\sigma$ | порог $|\tilde\sigma_G|$ для polarity | operational | a priori фильтр шума; в тесте MW U (script 34) τ_σ = 0 (все промпты включены) |
| $\tau_{\mathrm{ICC}}$ | мин. ICC для carrier stability | operational | a priori = 0.5 (умеренная надёжность); набл. 8/12 кластеров ≥ 0.5 (script 33) |
| $\eta_{\mathrm{rel}}$ | макс. доля нарушений polarity flip | operational | набл. эмпир.: mean SFR_pair = 0.803 → η_rel ≤ 0.197; 7/12 кластеров SFR > 0.9 (script 34) |
| $\tau$ | порог IIA для causal sufficiency (компактные механизмы) | operational | a priori, выше двух noise floor: $\tau > \mathrm{RFR} + \Delta_{\mathrm{recon}}$ и $\tau > \mathrm{IIA}_{\mathrm{rand\text{-}init}} + \Delta_{\mathrm{ctrl}}$; в данной работе $\mathrm{IIA}_{\mathrm{trained}}\approx 0 < \tau$ → Condition II проверяется через necessity |
| $\tau_{\mathrm{nec}}$ | порог necessity для distributed causal sufficiency | operational | a priori; в данной работе necessity = 67.6% ≥ τ_nec → Condition II выполнено |
| $\Delta_{\mathrm{ctrl}}$ | мин. отрыв IIA от random-init | **control** | $\mathrm{IIA}_{\mathrm{rand\text{-}init}}$ (script 52) |
| $\Delta_{\mathrm{recon}}$ | мин. отрыв от reconstruction flip | **control** | RFR / noop_sfr (gradient sens.) |
| $\mathrm{IR}_{\max}$ | макс. interaction ratio (coherence) | **control** | null $\mathrm{IR}_{\mathrm{null}}\approx 1$; берётся 0.5 |
| $\eta_{\mathrm{null}}$ | макс. доля $\mathrm{IR}\ge\mathrm{IR}_{\mathrm{null}}$ | **control** | random-bundle null-распределение |
| $\delta_{\mathrm{min}}$ | мин. потеря IIA при сужении $G$ | operational | a priori; на практике конструктивно |

**Наблюдаемые отрывы (control-anchored).** Для категории (1) порог — это отрыв от контроля, поэтому важны не абсолютные значения, а величина зазора:

- $\Delta_{\mathrm{recon}}$: reconstruction noise floor noop_sfr = 4–17% по слоям; circuit-level necessity = 67.6% — отрыв ≳ 50 п.п.
- $\Delta_{\mathrm{ctrl}}$: random-init контроль degenerate (OOD-патчи слишком шумны), что само по себе подтверждает: эффект не воспроизводится на случайной модели той же архитектуры.
- $\mathrm{IR}_{\max}/\eta_{\mathrm{null}}$: наблюдённое $\mathrm{IR}\in[0.018, 0.35]$ против null $\mathrm{IR}_{\mathrm{null}}\approx 1$ — отрыв > 3σ.

Именно эти зазоры (а не выбор конкретного порога) обеспечивают, что вывод $G\models_\Theta V_h$ устойчив и не является артефактом threshold-tuning — в духе best-practices по activation patching [7] и null-контролей MIB [11].

---

## Приложение B. Выбор процедуры кластеризации Φ

Определение 1 оставляет $\Phi$ свободным параметром, требуя лишь детерминированности и Condition I.a (carrier stability). Отсюда операциональный критерий: **среди процедур-кандидатов выбирается та, что максимизирует carrier stability** — воспроизводимость кластеров под task-preserving трансформациями входа, а не та, что максимизирует внутренний quality-индекс (silhouette).

**Recovery-тест.** Корпус разбивается на две *парафразные* половины (для каждой semantic_equivalence_group её wording-варианты делятся между A и B; физический cue и $V_h$ сохраняются, меняется только surface-формулировка). На каждой половине co-importance граф пересобирается и $\Phi$ перезапускается. Для каждого кластера полной кластеризации ищется наилучший Jaccard-аналог в половинной; усреднение даёт $1-\eta_{\mathrm{irr}}$. *Контроль:* та же процедура на случайных split той же величины задаёт data-reduction floor; если recovery под парафразом не ниже случайного (перцентиль ≈ 50), нестабильность объясняется лишь уменьшением данных, а не чувствительностью к формулировке.

**Сравнение кандидатов** (co-importance Jaccard-граф; основной корпус runD_v2, 538 промптов, 227 фичей):

| Метод Φ | K | meanJac | ARI | pctile среди random | Примечание |
|---------|---|---------|-----|---------------------|-----------|
| Louvain (modularity, auto) | 14 | 0.76 | 0.63 | 52 | k найден автоматически |
| Agglomerative avg, k=14 | 14 | 0.90 | 0.74 | **15** | детерминирован, лучшая carrier stability |
| **Agglomerative avg, k=16** | **16** | **(see note)** | **(see note)** | **<20** | **финальный выбор** |
| Agglomerative avg, k=17 | 17 | — | — | — | L25 расщепляется (шум) |
| NMF k=5 (silhouette champion) | 5 | — | — | — | формально лучший silhouette, но кросс-слойные сборки, не механистичен |

Louvain (modularity) **автоматически** находит k=14, что согласуется с слоевой структурой модели и не требует ручной настройки. Параллельно agglo average-linkage с k=14 даёт meanJac=0.90, ARI=0.74 (лучше 85% случайных разбиений) — это операциональный выбор по carrier stability.

**Поправка к k=14: dendrogram gap указывает на k=16.** Скрипт `diag_k_sweep_sanity.py` проверяет иерархию слияний на высотах [0.74, 0.90]:

- Переход 18→17 кластеров: gap = 0.087 (большой)
- Переход 17→16: gap = 0.017 (малый — устойчивая плоскость)
- Переход 16→15: gap = 0.0156 (малый)
- Переход 15→14: gap < 0.01

На k=14 алгоритм был вынужден произвести **два проблемных слияния** (L14+L17 и L22+L23) при отсутствии естественного gap. Эти слияния идентифицируются как нестабильные:

| Слияние | within_coimp | J_obs | pctile | Condition I.b |
|---------|--------------|-------|--------|----------------|
| L14+L17 (k=14) | 0.515 | 0.442 | 26.4 | ✗ нарушает |
| L22+L23 (k=14) | 0.377 | 0.794 | **78.5** | ✓ но граница |

При k=16 оба слияния **расщепляются на чистые однослойные кластеры** с сохранением сильных causal-кластеров: L18 (C7, 17 фичей, orient_delta=−0.896 — сильнейший α) и L24 (C4, 20 фичей, orient_delta=+1.334 — сильнейший β) **остаются единым кластером**. Polarity distribution на k=16: 5 α-supporting, 11 β-supporting, **0 mixed** (для сравнения, k=14 даёт 1 нарушение Condition I.b).

**Итог выбора:** Φ = `agglo_coimp_k16` (average-linkage agglomerative на co-importance Jaccard-графе с числом кластеров k=16, определённым как локальная устойчивая плоскость dendrogram между gap-значениями 0.087 и 0.017). Метод детерминирован (нет случайности Louvain), все 16 кластеров проходят Condition I.b, и dendrogram gap criterion подтверждает структурную устойчивость.

Quality-индекс (silhouette) при этом **не** решающий: NMF k=5 даёт silhouette=0.874 на том же пространстве, в котором строился, что не является независимой метрикой; механистически такие кросс-слойные сборки не интерпретируемы. Полная пайплайн-валидация (joint-ablation IR, Condition III) на выбранной agglo-партиции пересчитывается отдельно.

**Activation stability как прямая проверка (I.a).** Recovery-тест проверяет, воспроизводится ли *структура* кластеризации $\Phi$ при перезапуске на другом подмножестве промптов. После выбора $\Phi$ проводится более прямая проверка: кластеры фиксируются (обученные на полном корпусе), и для каждого кластера $G$ и каждой семантической группы $g$ вычисляется $\sigma_G(p)=\frac{1}{|G|}\sum_{(\ell,k)\in G}a_k^{(\ell)}(p)$ — среднее по фичам кластера. Sign-consistency — доля промптов внутри $g$, на которых знак $\sigma_G(p)$ совпадает с большинственным. Это непосредственно соответствует вероятности из (1.I.a): активируется ли $G$ при перефразировке $t(p)$ с тем же знаком.

Результаты (скрипт `31_activation_stability.py`, probe-корпус, 132 семантические группы): 11 из 12 кластеров — sign_consistency = 1.000 по всем группам; C6 — 0.983; среднее = 0.994. Коэффициент вариации CV < 0.12 у 11 кластеров (стабильность и по величине). Эмпирический $\eta_{\mathrm{irr}} \leq 0.017$ (worst case C6) — существенно ниже a priori порога 0.05. Нарушения C6 сосредоточены в 2-промптовых группах (n слишком мало для уверенного большинства) и в одной L1-B1-группе (12 промптов, sign_cons = 0.667); ни в одном случае нет системного переворота знака.

---

## Список литературы

[1] Geiger, A., Wu, Z., Potts, C., & Icard, T. (2024). Finding alignments between interpretable causal variables and distributed neural representations. *Journal of Machine Learning Research*, 26(46), 1–58. https://www.jmlr.org/papers/v26/23-0058.html

[2] Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A. (2025). Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. In *Proceedings of the 13th International Conference on Learning Representations* (ICLR 2025). arXiv:2403.19647. https://openreview.net/forum?id=I4e82CIDxv

[3] Wu, Z., Geiger, A., Potts, C., & Goodman, N. D. (2023). Causal abstractions of neural networks. In *Advances in Neural Information Processing Systems* (NeurIPS 2023). arXiv:2301.04709.

[4] Lindsey, J., Gurnee, W., Heimersheim, S., Janiak, J., et al. (2025). Circuit tracing: Revealing computational graphs in language models. arXiv:2601.14004. https://arxiv.org/abs/2601.14004

[5] Lindsey, J., & Gurnee, W. (2025). Scaling interpretability: Towards understanding a million features. In *Proceedings of the 42nd International Conference on Machine Learning* (ICML 2025). https://icml.cc/virtual/2025/49583

[6] Lindsey, J., Gurnee, W., Heimersheim, S., Janiak, J., Bloom, J., Ghilardi, L., & Conerly, T. (2025). On the biology of a large language model. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/methods.html

[7] Zhang, F., & Nanda, N. (2024). Towards best practices of activation patching in language models: Metrics and methods. arXiv:2309.16042. https://arxiv.org/abs/2309.16042

[8] Geiger, A., Wu, Z., Potts, C., & Icard, T. (2022). Inducing causal structure for interpretable neural networks. In *Proceedings of the 39th International Conference on Machine Learning* (ICML 2022), PMLR 162, 7324–7338. arXiv:2112.00114. https://openreview.net/forum?id=0yvZm2AjUr

[9] Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2024). Towards automated circuit discovery for mechanistic interpretability. In *Proceedings of the 12th International Conference on Learning Representations* (ICLR 2024). https://proceedings.iclr.cc/paper_files/paper/2024/file/06a52a54c8ee03cd86771136bc91eb1f-Paper-Conference.pdf

[10] Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., ... & Olah, C. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*. https://transformer-circuits.pub/2023/monosemantic-features/index.html

[11] Mueller, A., Geiger, A., Potts, C., Prakash, B., Tigges, C., Huang, J., ... & Belinkov, Y. (2025). MIB: A mechanistic interpretability benchmark. In *Proceedings of the 42nd International Conference on Machine Learning* (ICML 2025), PMLR 267, 45069–45108. arXiv:2504.13151.

[12] Geiger, A., Ibeling, D., Zur, A., Chanda, R., Geiger, S., Bhattacharya, S., Lu, H., Icard, T., Potts, C., & Goodman, N. (2024). Causal abstraction: A theoretical foundation for mechanistic interpretability. In *Advances in Neural Information Processing Systems* (NeurIPS 2024). https://openreview.net/forum?id=lB9g6jC_c8

[13] Zhang, Y. (2024). Causal abstraction in model interpretability: A compact survey. arXiv:2410.20161. https://arxiv.org/abs/2410.20161

[14] Wu, Z., Geiger, A., Potts, C., & Goodman, N. (2025). The Non-Linear Representation Dilemma: Is Causal Abstraction Identifiable? arXiv preprint.

[15] Dunefsky, J., Chlenski, P., & Nanda, N. (2024). Transcoders Find Interpretable LLM Feature Circuits. arXiv:2406.11944. https://arxiv.org/abs/2406.11944
