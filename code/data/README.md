#### Content explanation：

1. `AL-CPL`: This folder stores data from the four fields of the AL-CPL data set. Before the experiment, we performed certain processing on the data.

   `dm_concepts.csv`: This file stores the deduplicated conceptual data in the `data-mining` field.

   `ind.data-mining.test.index`: This file stores the index value corresponding to the concept after deduplication in the `data-mining` field.

   `data-miningConcepts.xlsx`: This file stores concept pair data, label values and semantic similarity values. Each concept is represented by the corresponding index value in the file `ind.data-mining.test.index`.

   The other three data field files are similar to the data content stored in the above `data-mining` field.

2. `BERT`: Stored in this folder are BERT-based concept feature vectors and cosine similarities.

   `feature_vector`: Stored in this folder are 768-dimensional feature vectors of four domain concepts and cosine similarity based on BERT.

   `concept content preservation`: This file stores the content of the English Wikipedia pages corresponding to each concept in the four fields.

   `300 words before concept`: Stored in this folder are the first 300 words corresponding to the content of the concept page.

   `contentSave.py`: This code file obtains the English Wikipedia page content corresponding to the concept.

   `Get the first 300 words of the concept.py`: This code file fetches the first 300 words of concept content.

   `Bert-300.py`: Input the first 300 words of the concept into BERT to obtain the 768-dimensional feature vector of the concept.

