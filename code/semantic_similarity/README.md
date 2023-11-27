#### Content explanation：

1. `BERT`: Stored in this folder are BERT-based cosine similarities in four fields.
2. `PMI`: Stored in this folder are the `PMI` values of data in four fields.
3. `files`: This folder stores the semantic relevance after summarizing the data in the four fields. The semantic relevance is calculated based on BERT's cosine similarity and PMI value. Here, we assign a weight of 0.5 to each and take their final sum as the final semantic relevance value.
4. `wLabel(PMI+BERT)`: Stored in this file are data based on different experimental settings. `wLabel` constructs a concept map based only on weak labels. `Label_-half` is to update the concept map by removing half of the wrong labels based on the original weak labels. `Label_-all` updates the concept map by removing all wrong labels based on the original weak labels. `Label_15%` is to update the concept map by adding another 15% of the real label concept pair data based on the original weak labels. `Label_30%` is to update the concept map by adding another 30% of the real label concept pair data based on the original weak labels.
5. `dm_train.xlsx`: Training set data.
6. `dm_test.xlsx`: Test set data.
7. `ind.data-mining.graph`: Constructed concept map data.
8. `ind.data-mining.up.graph`: Upper triangular concept map data based on the concept map `ind.data-mining.graph`.
9. `ind.data-mining.down.graph`：Lower triangular concept map data based on the concept map `ind.data-mining.graph`.
10. `ind.data-mining02.graph`: Really labeled concept map data based on the concept graph `ind.data-mining.graph`.
11. `ind.data-mining02.up.graph`: Upper triangle concept map data based on concept map `ind.data-mining02.graph`.
12. `ind.data-mining02.down.graph`：Lower triangle concept map data based on concept map `ind.data-mining02.graph`.

