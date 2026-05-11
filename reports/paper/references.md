# References

Five core references carried into the final paper. Bibkeys below are
the LaTeX citation keys; each entry includes a one-line note on how
it is used in our text. The `.bib` file built from this list at the
LaTeX-assembly pass is `references.bib`.

## arik2021tabnet

Arik, S. Ö., & Pfister, T. (2021). *TabNet: Attentive Interpretable
Tabular Learning.* Proceedings of the AAAI Conference on Artificial
Intelligence, 35(8), 6679–6687.

**Cited in:** §2 (positioning), §3.1 (architecture description),
§5.2 (hyperparameter conventions), §10.2 (mask interpretation),
§11.1 (selection-sparsity mechanism claim).

## chen2016xgboost

Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting
System.* Proceedings of the 22nd ACM SIGKDD International Conference
on Knowledge Discovery and Data Mining, 785–794.

**Cited in:** §2 (positioning), §3.2 (baseline rationale),
§5.1 (hyperparameter conventions), §10.1 (gain importance),
§11.1 (split-finding-is-near-exhaustive mechanism claim).

## grinsztajn2022why

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do
tree-based models still outperform deep learning on typical tabular
data?* Advances in Neural Information Processing Systems 35,
507–520.

**Cited in:** §2 (debate positioning), §3.3 (mixed-verdict
synthesis).

## shwartz2022tabular

Shwartz-Ziv, R., & Armon, A. (2022). *Tabular data: Deep learning
is not all you need.* Information Fusion, 81, 84–90.

**Cited in:** §2 (debate positioning), §3.3 (no-tabular-DL-wins-
consistently claim), §11.1 (low-feature regime is least favorable
for attention).

## guo2017calibration

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On
Calibration of Modern Neural Networks.* Proceedings of the 34th
International Conference on Machine Learning, 1321–1330.

**Cited in:** §3.4 (calibration literature), §8.1 (ECE definition
and ten-bin convention), §8.3 (temperature-scaling protocol),
§12 (binning convention pre-fixed), §13 (methodological caution
phrased downstream of Guo).

## Bibkey-to-citation map for the LaTeX pass

| location in paper | bibkey |
| --- | --- |
| §2 paragraph 1 (XGBoost as de-facto baseline) | chen2016xgboost |
| §2 paragraph 1 (TabNet as attention alternative) | arik2021tabnet |
| §2 paragraph 1 (Grinsztajn mixed-verdict) | grinsztajn2022why |
| §2 paragraph 1 (Shwartz-Ziv caution) | shwartz2022tabular |
| §3.1 (TabNet architecture) | arik2021tabnet |
| §3.2 (XGBoost baseline) | chen2016xgboost |
| §3.3 (debate synthesis) | grinsztajn2022why, shwartz2022tabular |
| §3.4 (calibration thread) | guo2017calibration |
| §8.1 (ECE convention) | guo2017calibration |
| §8.3 (temperature-scaling protocol) | guo2017calibration |
| §10.2 (mask interpretation) | arik2021tabnet |
| §11.1 (low-feature regime mechanism) | shwartz2022tabular |
