# XAI-Analytics: project assessment and recommended direction

Assessment date: 24 October 2026. Primary objective: a respected open-source project and a strong engineering portfolio. Commercialization is a secondary option to validate later.

**Recommendation: invest in one bounded modernization release, focused on reproducible explanation comparison and stability for tabular classifiers.** Preserve the thesis work as a historical research artifact. Reassess continued investment after external users try the new release.

## What the project is today

XAI-Analytics is a notebook-based workbench for loading data, preprocessing it, training several models, and comparing global and local explanations. The main notebook supplies an ipywidgets interface; Python utility modules implement training, explanations, comparison, and display. Examples cover Adult Census Income and cervical-cancer risk factors.

The valuable work includes:

- An end-to-end educational workflow, including explanations of the underlying concepts.
- Comparisons across classifiers and explanation methods, rather than isolated plots.
- Rank-biased overlap (RBO) comparisons of feature rankings, including randomized baselines.
- LIME stability measurements and kernel-width optimization experiments.
- Existing examples, rendered outputs, a demonstration video, and a BSD-3-Clause license.

The original purpose remains useful. However, the README's broad promise of making model interpretation easy is insufficiently specific to distinguish the project today. The strongest potential identity is the evaluation and comparison workflow already emerging from the research.

Suggested positioning:

> XAI-Analytics helps you compare and stress-test explanations of tabular machine-learning models, with reproducible experiments and shareable reports.

Primary users: Python data scientists and applied researchers working with scikit-learn-compatible tabular classifiers. Secondary users: instructors teaching interpretability. A concrete job to solve is: “My model predicts well, but its explanation changes between runs or disagrees with another method. Help me measure that and document what happened.”

## Evidence and review boundaries

Reviewed the tracked source modules, README, dependency files, license, main and example notebook content, local changes, Git history, public repository metadata, and official documentation for alternatives. The local HEAD is `f68b9b8998123cab521041bc05dc8e0d3d05b3cb`, dated 17 June 2023. This is the checked-out history, not a claim that every remote ref was synchronized.

The checkout contains substantial pre-existing modified and untracked experiments. Findings below distinguish those from HEAD. All five tracked Python modules in HEAD pass an AST syntax check. The working copy of `util/commons.py` fails that check. The existing virtual environment cannot start its configured Python 3.9 interpreter. A complete application run and fresh dependency installation were not performed; saved notebook output is not evidence of current reproducibility. Three inspected committed example/application notebooks contain no saved error outputs, while the modified Income notebook does.

No application code was changed during this assessment. Public web pages can be cached; repository popularity and vendor claims do not establish product-market fit.

## Priority technical findings

| Priority | Finding and evidence | Consequence and recommended action |
|---|---|---|
| P0 | Working-copy `util/commons.py:1419` has an empty `for process in n_jobs` loop. AST parsing raises `IndentationError` at line 1422. This is absent from HEAD. | The main utility module cannot import. Preserve the experiments and restore an executable baseline before combining them with release work. |
| P0 | Working-copy `util/commons.py:1835` overwrites the selected prediction with `prediction = 0`. This is absent from HEAD. | A local SHAP explanation can describe a different class from the selected/predicted one. Make output selection explicit and test class labels and indices. |
| P1 | `util/commons.py:275` resamples the outer training partition before `train_model` passes it to cross-validation. Present in HEAD. | Inner validation folds can be influenced by samples used to generate synthetic training data. Put resampling inside a pipeline fitted separately in each CV fold. The outer test partition is separated first; this finding does not establish that every reported outer-test score is contaminated. |
| P1 | `util/commons.py:209` fits preprocessing to both train and test to compare category sets, and retries splits until they match. Present in HEAD. | The holdout composition influences split selection; rare categories may trigger repeated failures. Fit on training data and transform the holdout with a stable feature schema. The subsequent training pipeline refits, so this is not simply a claim that final scaling parameters come from the test data. |
| P1 | `util/model.py:222` builds the SHAP background from the test set, reduced to one centroid; LIME initialization around line 289 also uses test data as its reference distribution. Present in HEAD. | Make background/reference data an explicit, documented choice. Default to a representative training reference, freeze it for comparisons, and report sensitivity to it. Using a test-derived explanation reference is not automatically predictive training leakage, but it changes what the explanation means. |
| P1 | `util/commons.py:451` retains only positive ELI5 weights, while the SHAP path uses absolute contributions. RBO compares resulting key orders. Present in HEAD. | Strong negative coefficients can disappear from one ranking. Define magnitude versus signed comparisons, feature grouping, class, ties, and truncation consistently before drawing conclusions from agreement. |
| P1 | `util/commons.py:1067` calls SHAP `dependence_plot` under a PDP function and UI category. Present in HEAD. | SHAP contribution-versus-feature plots and prediction-based PDPs have different meanings. Label and explain them separately. |
| P1 | `util/commons.py:1189` reconstructs categories by splitting encoded feature names on underscores. Present in HEAD. | Features/categories containing underscores and dropped binary categories can be misrepresented. Derive mappings from fitted encoder metadata rather than parsing display names. |
| P1 | `util/dataset.py:241` disables TLS verification by default; line 259 uses `200 >= status < 300`. Present in HEAD. | The predicate accepts 100 and rejects 201/204, confirmed with a small direct check. Enable certificate verification, use `200 <= status < 300`, and add timeouts and bounded downloads. |
| P2 | Regression appears in model types and training code despite being a README non-goal; task inference treats numeric multiclass targets as regression, and training search uses accuracy. | Publish a precise support matrix and require explicit task selection. Start with validated binary tabular classification; expand only with matching tests. |
| P2 | The working `commons.py` exceeds 2,100 lines and combines training, statistics, plotting, UI-facing text, and optimization. Model objects also hold widgets. | Extract a small testable computation package, with notebooks as consumers. Avoid starting with a full web application rewrite. |

The CV recommendation follows [imbalanced-learn's documented leakage guidance](https://imbalanced-learn.org/stable/common_pitfalls.html). The plotting distinction follows the [SHAP dependence-plot documentation](https://shap.readthedocs.io/en/latest/generated/shap.plots.scatter.html). Training-only preprocessing follows [scikit-learn's common-pitfall guidance](https://scikit-learn.org/stable/common_pitfalls.html).

Installation is another release blocker. The README targets Python 3.7–3.9, all now unsupported according to the [Python version lifecycle](https://devguide.python.org/versions/). Dependency pins largely date to 2021. Installation requires bypassing dependency resolution and patching installed Skater files. Some required imports are excluded or commented out in installation inputs. A fresh supported environment needs to become a tested deliverable, not an undocumented local workaround.

## Where the project fits

| Alternative | Existing offering | Implication |
|---|---|---|
| [ExplainerDashboard](https://github.com/oegedijk/explainerdashboard/blob/master/docs/source/index.rst) | Interactive explanations, SHAP, permutation importance, PDPs, performance views. | A convenient visualization wrapper alone is a weak distinction. |
| [InterpretML](https://interpret.ml/docs/) | Interpretable models, black-box explainers, global/local visualizations. | Integrate established methods and demonstrate a focused workflow advantage. |
| [DALEX](https://dalex.drwhy.ai/) | Model explanation, comparison, fairness, and interactive analysis. | Comparison itself is not an uncontested niche. |
| [Quantus](https://github.com/understandable-machine-intelligence-lab/quantus/) | Quantitative explanation evaluation across multiple metric categories, principally around neural-network workflows, with tabular support too. | Do not claim to invent explanation evaluation. Investigate reuse and make scikit-learn tabular usability and reproducible reporting the differentiator. |
| [Evidently](https://www.evidentlyai.com/) and [Fiddler](https://www.fiddler.ai/) | Broader evaluation/monitoring platforms; Fiddler also advertises commercial deployment and governance capabilities. | Avoid competing across the entire monitoring platform surface. A focused local evaluation workflow can be complementary. |

These sources demonstrate an established problem area and substantial competition. They do not prove demand for another product. The proposed gap—an approachable, rigorous explanation-reliability workflow for tabular Python models—is a hypothesis to test with users.

## The next useful release

Build one complete workflow: choose a small example or pass a fitted pipeline, select evaluation/reference data, generate explanations for explicit model outputs, repeat under controlled settings, and export a report.

Start with SHAP, LIME, and scikit-learn permutation importance. Use [scikit-learn's inspection tools](https://scikit-learn.org/stable/inspection.html) for PDP/ICE. Evaluate removing Skater and PDPBox from the default dependency path. Keep XGBoost and the notebook UI optional where practical. The initial Python target should be a small supported matrix such as 3.12/3.13, subject to actually verifying dependency compatibility.

The distinctive report should show:

- Model performance and evaluation population, with clear class/output labels.
- Explanation variation across independent random seeds.
- Top-k agreement and RBO, with documented persistence, depth, tie handling, and randomized baselines.
- Sensitivity to reference data, sampling budgets, and LIME kernel width.
- Local surrogate fidelity alongside stability. A stable explanation can still be inaccurate.
- Runtime and model-call budgets.
- Model/data fingerprints, preprocessing, dependency versions, seeds, and full evaluation configuration.

Report each property separately. Agreement is not correctness; stability is not faithfulness; explanations do not establish causality or fairness. Correlated features and implausible perturbations should be explicit limitations. For kernel optimization, evaluate on new seeds or held-out perturbation samples to avoid reporting the same score used to tune the kernel as independent evidence.

Use synthetic datasets with known generating rules, irrelevant features, correlated features, and controlled failure cases to test behavior. Supplement those with a real tabular example whose provenance and permitted use are documented. Preserve the medical example as historical educational material; it should not imply clinical validation.

## Making the GitHub project credible

The tracked snapshot has a README and license, but no `.gitignore`, package metadata, automated test suite, or tracked CI workflows. Prioritize reproducibility and scientific correctness before cosmetic repository additions.

1. Preserve the thesis version with a historical release/tag when ready, including environment details and a thesis citation/link if available. Curate local experiments individually; do not delete or commit them wholesale.
2. Package computation under `src/xai_analytics/`, with `pyproject.toml`, explicit dependencies/extras, a repeatable development environment, and bounded support promises. Separate explainers, evaluation, reporting, and notebook UI.
3. Add regression tests for the findings above. Include unseen categories, dropped binary categories, feature names with underscores, output-class mapping, fixed-seed reproducibility, and ranking semantics.
4. Add CI for a clean installation, imports, tests, formatting/linting, and one tiny offline notebook/report smoke test. Test the built distribution, not only the checkout. Keep expensive research experiments outside the default CI path.
5. Write a README centered on one user problem, a screenshot/report, a verified quickstart, exact supported inputs, limitations, and a comparison with alternatives. Update outdated TODOs: some preprocessing features already exist.
6. Add contribution instructions, bug/feature templates, security reporting instructions, a code of conduct appropriate to a small community, a changelog, citation metadata, and a short public roadmap. [GitHub documents these community-health conventions](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/about-community-profiles-for-public-repositories).
7. Publish versioned documentation and releases; show badges only for checks that run. Move large rendered examples to documentation or release artifacts when practical. Add ignore rules for virtual environments, caches, IDE files, and logs.
8. Publish a technical case study showing a real explanation failure, how the tool detects it, and its limitations. Invite users to reproduce the result and report failures. This is stronger portfolio evidence than a long unsupported feature list.

A good initial success criterion is that another person can install the package, generate the example report, understand its claims, and add a small contribution without the author's intervention.

## Commercial options

The existing [BSD-3-Clause license](https://opensource.org/license/bsd-3-clause) permits commercial use subject to its conditions. Before selling a distribution or hosted service, check the provenance and licenses of included dependencies, datasets, and copied research code, and resolve any relevant university ownership agreement. The license itself does not create an exclusive commercial advantage.

| Option | Fit now | What a customer would pay for |
|---|---|---|
| Paid workshops and technical model-review engagements | Best first experiment | Expertise, setup, interpretation, a useful report, and saved analyst time. |
| Paid integration/private deployment support | Plausible after external adoption | Connecting the tool to existing pipelines and supporting a dependable environment. |
| Hosted report history and team review | Later, conditional | Scheduled runs, version comparisons, access control, shared annotations, and retained evidence. |
| Broad self-service XAI SaaS | Weak initial bet | Requires differentiation plus secure execution, operations, support, and customer acquisition. |
| Sponsorship or research/teaching support | Compatible with the open-source goal | Sustainability rather than predictable product revenue. |

For a later service, keep a useful local evaluation engine open source. First test a design where computation remains in the customer's environment and selected report artifacts are shared; reports can still contain sensitive information and need appropriate controls. A service executing uploaded Python model files introduces a much larger isolation and security problem than hosting reports.

Do not build subscriptions because the software can be hosted. First establish a recurring task: for example, a team repeats a model review on each release and needs comparisons, review history, and collaboration. A one-off explanation task may be better served by a library or paid engagement.

Validate with approximately 10 conversations across practitioners, researchers, and instructors. Ask about the last real model-review task, current tools, time spent, failures, and the person who owns budget. Then watch 3–5 users attempt the workflow on their own problem. Ask for a paid pilot only after the tool delivers a concrete benefit. No pricing, revenue, market size, or willingness to pay has been established by this assessment.

## A bounded 90-day plan

This is a sequencing proposal, not an effort estimate. Adjust calendar time to available weekly hours.

| Period | Deliverable | Evidence to proceed |
|---|---|---|
| Days 1–14 | Preserve historical work, choose supported scope, resolve executable/dependency baseline, establish minimal CI. | A clean checkout installs and a tiny classification example runs without manual package patches. |
| Days 15–35 | Correct evaluation semantics, separate core computation, test known failure cases. | Class selection, feature alignment, fold-local resampling, and deterministic configurations are covered by tests. |
| Days 36–60 | Release one explanation-comparison/stability workflow and HTML/JSON report with provenance. | An unfamiliar user produces and explains a report independently. |
| Days 61–90 | Publish docs and a reproducible case study, run user sessions, improve onboarding, release a beta. | Several external users complete the workflow; at least some return or request concrete improvements. |

Suggested decision gate: after the first usable release, recruit five external trial users and look for at least two repeat users or one recurring teaching/research use. These are chosen learning targets, not industry benchmarks. If users value the comparisons, deepen the reliability work. If teaching dominates, focus on workshops and course materials. If repeated professional reviews emerge, explore paid support and then hosting. If there is little use after direct testing and onboarding fixes, preserve a polished research release and reduce maintenance commitments.

For the portfolio objective, the highest-value outcome is a coherent package with sound methodology, readable architecture, useful tests, documented tradeoffs, reproducible examples, and evidence that other people can use it. Large scale or SaaS revenue is not required to demonstrate those qualities.
