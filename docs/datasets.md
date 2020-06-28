# Datasets

Here is something about classification datasets.

It will be much easier to preprocess and load data via [torchtext](https://github.com/pytorch/text), check out its documentation [here](https://pytorch.org/text/).

Here are statistics of some popular classification datasets:

| Dataset                | Classes | Train Samples | Test Samples | Total     | Download                                                     |
| ---------------------- | ------- | ------------- | ------------ | --------- | ------------------------------------------------------------ |
| AG News                | 4       | 120,000       | 7,600        | 127,600   | [Goole Drive](https://drive.google.com/file/d/0Bz8a_Dbh9QhbUDNpeUdjb0wxRms/view?usp=sharing) |
| Sogou News (Chinese)   | 5       | 450,000       | 60,000       | 510,000   | [Goole Drive](https://drive.google.com/file/d/0Bz8a_Dbh9QhbUkVqNEszd0pHaFE/view?usp=sharing) |
| DBpedia                | 14      | 560,000       | 70,000       | 630,000   | [Goole Drive](https://drive.google.com/file/d/0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k/view?usp=sharing) |
| Yelp Review Polarity   | 2       | 560,000       | 38,000       | 598,000   | [Goole Drive](https://drive.google.com/file/d/0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg/view?usp=sharing) |
| Yelp Review Full       | 5       | 650,000       | 50,000       | 700,000   | [Goole Drive](https://drive.google.com/file/d/0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0/view?usp=sharing) |
| Yahoo Answers          | 10      | 1,400,000     | 60,000       | 1,460,000 | [Goole Drive](https://drive.google.com/file/d/0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU/view?usp=sharing) |
| Amazon Review Full     | 5       | 3,000,000     | 650,000      | 3,650,000 | [Goole Drive](https://drive.google.com/file/d/0Bz8a_Dbh9QhbZVhsUnRWRDhETzA/view?usp=sharing) |
| Amazon Review Polarity | 2       | 3,600,000     | 400,000      | 4,000,000 | [Goole Drive](https://drive.google.com/file/d/0Bz8a_Dbh9QhbaW12WVVZS2drcnM/view?usp=sharing) |
| IMDB                   | 2       | 25,000        | 25,000       | 50,000    | [Link](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) |
| SST-2                  | 2       | /             | /            | 94.2k     | [Link](http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip) |
| SST-5                  | 5       | /             | /            | 56.4k     | [Link](http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip) |
| TREC                   | 6 / 50  | 5,452         | 500          | 5,952     | [Link](https://cogcomp.seas.upenn.edu/Data/QA/QC/)           |

&nbsp;

## Text Classification

All of the following datasets can be downloaded [here](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) (Google Drive). They are proposed and described in this paper:

[**Character-level Convolutional Networks for Text Classification.**](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) *Xiang Zhang, et al.* NIPS 2015.

&nbsp;

- **AG News**

  News articles, original data are from [here](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html).

  **4 Classes:** 0: World, 1: Sports, 2: Business, 3: Sci/Tech

- **Sogou News (Chinese)**

  News articles from [ SogouCA](https://www.sogou.com/labs/resource/ca.php) and [SogouCS](https://www.sogou.com/labs/resource/cs.php) (manually labeled using URLs).

  **5 Classes:** 0: Sports, 1: Finance, 2: Entertainment, 3: Automobile, 4: Technology

- **DBpedia**

  Title and abstract of each Wikipedia article, original data are from [here](https://wiki.dbpedia.org/services-resources/datasets/dbpedia-datasets).

  **14 Classes:** 0: Company, 1: Educational Institution, 2: Artist, 3: Athlete, 4: Office Holder, 5: Mean Of Transportation, 6: Building, 7: Natural Place, 8: Village, 9: Animal, 10: Plant, 11: Album, 12: Film, 13 : Written Work

- **Yelp Review Full**

  Reviews on Yelp, from Yelp Dataset Challenge 2015. [Here](https://www.yelp.com/dataset) is Yelp Dataset's homepage.

  **5 Classes:**  five levels of ratings from 0-4 (higher is better)

- **Yelp Review Polarity**

  Modified from Yelp Review Full, by considering stars 1, 2 negative, and 3, 4 positive.

  **2 Classes:** 0: Negative polarity, 1: Positive polarity

- **Yahoo Answers**

  Question title, question content and best answer from [Yahoo! Answers Comprehensive Questions and Answers version 1.0](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11).

  **10 Classes:** 0: Society & Culture, 1: Science & Mathematics, 2: Health, 3: Education & Reference, 4: Computers & Internet, 5: Sports, 6: Business & Finance, 7: Entertainment & Music, 8: Family & Relationships, 9: Politics & Government

- **Amazon Review Full**

  Reviews from Amazon, including title and content, original data are from [here](https://snap.stanford.edu/data/web-Amazon.html).

  **5 Classes:**  five levels of ratings from 0-4 (higher is better)

- **Amazon Review Polarity**

  Modified from Amazon Review Full, by considering stars 1, 2 negative, and 3, 4 positive.

  **2 Classes:** 0: Negative polarity, 1: Positive polarity

&nbsp;

## Sentiment Analysis

- [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/)

  Proposed in paper:

  [**Learning Word Vectors for Sentiment Analysis.**](https://www.aclweb.org/anthology/P11-1015.pdf) *Andrew L. Maas, et al.* ACL 2011.

  **2 Classes:** Negative, Positive

  **samples:** train: 25,000, test: 25,000

  **Description:** Movie reviews, the ratings range from 1-10. A negative review has a score ≤ 4, and a positive review has a score ≥ 7.

- [SST](https://nlp.stanford.edu/sentiment/)

  Movie reviews.

  - SST-5 (Fine-grained)

    **5 Classes:** Very Negative, Negative, Neutral, Positive, Very Positive

    **samples:**  94.2k

  - SST-2 (Binary)

    **2 Classes:** Negative, Positive

    **samples:**  56.4k

    **Description:**  Same as SST-5 but with neutral reviews removed and binary labels.

&nbsp;

## Question Classification

- [TREC](https://cogcomp.seas.upenn.edu/Data/QA/QC/)

  A dataset for classifying questions into semantic categories.

  **samples: ** train: 5,452, test: 500

  - TREC-6

    **6 Classes:** Abbreviation, Description, Entity, Human, Location, Numeric Value

  - TREC-50 (Fine-grained)

    **50 Classes**