# DISSETRAÇÃO

# Read-me: EU Migration Speech Evolution — Complete Text-Mining Pipeline

## 0. Project objective

### Research question

> **How did political parties evolve their speeches concerning major migration spikes over time?**

---

# 1. Data understanding / exploration

The first stage is to understand **what your observations actually represent** before doing any NLP.

## 1.1 Load the data

Start by:

* loading the dataset
* checking dimensions
* inspecting data types
* identifying categorical variables
* examining missing values
* checking duplicated rows
* checking unique values
* checking date ranges



| Variable     | Role                         |
| ------------ | ---------------------------- |
| `doc_id`     | Speech/document identifier   |
| `date`       | Speech date                  |
| `year`       | Speech year                  |
| `country`    | Country                      |
| `party_name` | Political party              |
| `epg_short`  | European Parliamentary Group |
| `language`   | Original language            |
| `speech`     | Original speech              |
| `speech_en`  | English speech               |
| `agenda`     | Agenda/category              |
| `gender`     | Speaker gender               |
| `birth_year` | Speaker birth year           |

---

## 1.2 Understand the temporal coverage

Determine:

* earliest speech
* latest speech
* number of speeches per year
* number of speeches per month
* number of speeches per country
* number of speeches per party



### Important question

You need to establish whether the dataset covers the periods you want to call **migration spikes**.

A spike should not simply mean "a year with lots of speeches." Ideally, it should correspond to an **external measure of migration/asylum arrivals**.

This distinction is crucial:

> Migration spike ≠ spike in the number of migration-related speeches.

You want an external migration indicator to identify the events/periods, and then examine speeches around them.

---

# 2. Define the migration spikes

This should be a separate methodological stage because it determines the treatment/control periods in your analysis.

## 2.1 Obtain migration data

Depending on your exact research question, you could use measures such as:

* asylum applications
* irregular border crossings
* arrivals
* migrant detections
* first-time asylum applicants
* migration-related EU statistics

Ideally, obtain **monthly data** because your speech data contains `datemonth`.

The migration dataset should have something like:

| month   | migration_measure |
| ------- | ----------------: |
| 2014-01 |               ... |
| 2014-02 |               ... |
| ...     |               ... |
| 2015-09 |               ... |

---

## 2.2 Define what counts as a spike

Do not arbitrarily select dates because they are famous migration events.

Define an operational rule.

For example:

### Option A — percentile

A spike is a month where migration is above the 90th percentile:

$$
Spike_t =
\begin{cases}
1 & \text{if Migration}_t > P_{90}\\
0 & otherwise
\end{cases}
$$

### Option B — standard deviation

Define a spike as:

$$
Migration_t > \mu + 2\sigma
$$

### Option C — year-on-year increase

Identify months where migration increases substantially relative to the previous year.

The choice should be theoretically justified.

---

## 2.3 Create event windows

Once spikes are identified, construct windows around them.

For example:

```text
Before spike     Spike period       After spike
   -3 months       0 months          +3 months
```

You could create:

* `pre_spike`
* `spike`
* `post_spike`

or a continuous variable:

```text
months_from_spike
```

For example:

| date    | spike_id | period | months_from_spike |
| ------- | -------- | ------ | ----------------: |
| 2015-05 | S1       | pre    |                -4 |
| 2015-06 | S1       | pre    |                -3 |
| 2015-09 | S1       | spike  |                 0 |
| 2015-10 | S1       | post   |                +1 |

This becomes extremely useful later.

---

# 3. Data preprocessing

Now clean the speech corpus.

## 3.1 Remove problematic observations

Check for:

* missing speech text
* empty speeches
* duplicate documents
* extremely short speeches
* speeches with only procedural text
* missing party
* missing date
* missing country

For example:

```python
df = df.dropna(subset=["speech_en", "date", "party_name"])
```

But **don't automatically delete observations with missing party/country**. First quantify how much data would be lost.

---

# 4. Text preprocessing

Create a clean version of the speech.

I recommend keeping **at least three versions**:

```text
speech_original
speech_clean
speech_model
```

Why?

Because aggressive preprocessing can destroy meaningful political language.

---

## 4.1 Basic normalization

Perform:

* lowercase
* whitespace normalization
* removal of HTML
* removal of URLs
* removal of obvious formatting artifacts
* handling of encoding problems

Example:

```python
text = text.lower()
```

---

## 4.2 Remove irrelevant material

Potentially remove:

* speaker metadata
* parliamentary formatting
* page numbers
* procedural markers
* repeated headers
* transcription artifacts

But be careful with:

> "migration", "migrant", "refugee", "asylum"

These are substantively important and must obviously remain.

---

## 4.3 Tokenization

Convert:

```text
"Migration policy needs reform"
```

into:

```text
["migration", "policy", "needs", "reform"]
```

---

## 4.4 Stopword removal

Remove generic words such as:

```text
the
and
of
to
in
```

But consider creating **two corpora**:

### Corpus A — standard NLP

Stopwords removed.

### Corpus B — linguistic analysis

Stopwords retained.

This matters because grammatical changes can themselves be politically meaningful.

---

## 4.5 Lemmatization

Convert:

```text
migrants
migrant
migration
```

into appropriate lexical forms where useful.

Use lemmatization rather than aggressive stemming for political text.

---

## 4.6 Preserve important multi-word expressions

This is particularly important for your topic.

You don't want:

```text
European Union
asylum seeker
illegal migration
border control
human rights
```

to be completely destroyed by preprocessing.

Create bigrams/trigrams such as:

```text
"asylum seeker"
"border control"
"illegal migration"
"European Union"
"fundamental rights"
```

---

# 5. Exploratory text analysis

Before modelling, understand what the corpus looks like.

Calculate:

* number of speeches
* number of words
* average speech length
* vocabulary size
* most frequent words
* most frequent bigrams
* vocabulary by party
* vocabulary by period
* vocabulary around migration spikes

---

## 5.1 Speech length

Examine whether speech length changes around migration spikes.

For example:

$$
SpeechLength_{it}
$$

where \(i\) is the speech and \(t\) is time.

This is a useful control variable later.

---

## 5.2 Migration vocabulary

Build a migration-related dictionary.

For example:

```text
migration
migrant
refugee
asylum
border
deportation
integration
immigration
emigration
citizenship
```

Then calculate:

$$
MigrationShare =
\frac{\text{migration-related words}}
{\text{total words}}
$$

This gives you a simple measure of **how much attention parties devote to migration**.

---

# 6. Feature extraction / vectorization

This is where you convert text into numerical representations.

I would **not rely on a single method**.

Use several complementary representations.

---

# 6.1 TF-IDF

Start with TF-IDF.

TF-IDF represents how distinctive words are within documents.

Conceptually:

$$
TFIDF(t,d)=TF(t,d)\times IDF(t)
$$

Use:

* unigrams
* bigrams
* potentially trigrams
* minimum document frequency
* maximum document frequency

Example:

```python
TfidfVectorizer(
    ngram_range=(1,2),
    min_df=5,
    max_df=0.9
)
```

TF-IDF is useful for:

* clustering
* classification
* identifying distinctive vocabulary
* comparing parties

---

# 6.2 Count vectors

Also create document-term matrices using word counts.

These are useful for:

* topic modelling
* word frequencies
* dictionary methods

---

# 6.3 Word embeddings

For a more advanced analysis, represent speeches using embeddings.

Possible approaches include:

* Word2Vec
* FastText
* Sentence-BERT
* other transformer embeddings

Instead of representing:

```text
"refugee"
```

as simply one column, embeddings represent semantic relationships.

This is particularly valuable because political language can change while retaining similar meaning.

---

# 6.4 Sentence/document embeddings

For your research question, I would strongly consider **document-level embeddings**.

Each speech becomes a vector:

$$
Speech_i \rightarrow \mathbf{x_i}
$$

Then you can measure:

$$
Similarity(Speech_i, Speech_j)
$$

This allows you to examine whether parties' speeches become:

* more similar
* more different
* closer to other parties
* further away from their own previous rhetoric

---

# 7. Dictionary-based features

Because your research question concerns **migration discourse**, dictionaries are particularly useful.

Create theoretically motivated dictionaries such as:

### Migration

```text
migration
migrant
immigrant
refugee
asylum
...
```

### Border/security

```text
border
control
security
illegal
deportation
crime
...
```

### Humanitarian/human rights

```text
humanity
humanitarian
rights
protection
solidarity
...
```

### Integration

```text
integration
inclusion
community
employment
education
...
```

### Economic

```text
labour
jobs
wages
economy
workers
...
```

Then calculate the prevalence of each category per speech.

This gives you interpretable variables such as:

```text
migration_intensity
security_intensity
humanitarian_intensity
integration_intensity
economic_intensity
```

---

# 8. Sentiment and emotion

You can also extract:

* sentiment
* positive/negative language
* fear
* anger
* trust
* anxiety
* hostility

However, **be careful with generic sentiment models**.

Political speeches are not ordinary consumer reviews.

A negative sentiment score may reflect discussion of:

> "the terrible conditions faced by refugees"

rather than negative attitudes toward refugees.

Therefore, sentiment should be treated as a **secondary feature**, not your primary measure of political position.

---

# 9. Topic modelling

This should probably be one of the central components of your project.

## 9.1 LDA

Run Latent Dirichlet Allocation.

The model identifies groups of words that frequently occur together.

For example, you might discover topics such as:

```text
Topic 1:
border, security, illegal, control, protection

Topic 2:
refugee, humanitarian, asylum, rights, protection

Topic 3:
integration, employment, education, society

Topic 4:
EU, Turkey, agreement, cooperation, external
```

You should **not assume these topics beforehand**.

Interpret them after estimating the model.

---

# 9.2 Choose the number of topics

Test different values:

```text
K = 5
K = 10
K = 15
K = 20
K = 25
```

Compare:

* topic coherence
* topic diversity
* interpretability

Then select a theoretically meaningful model.

---

# 9.3 Dynamic topic analysis

This is particularly relevant to your question.

You don't just want:

> What topics exist?

You want:

> **How does the prevalence of topics change over time and around migration spikes?**

For every speech, obtain:

$$
TopicProbability_{i,k}
$$

Then aggregate:

$$
TopicShare_{party,t,k}
$$

This gives you graphs such as:

```text
Topic: Border Security

2014 ───────╮
2015        ╰───────╮
2016                ╰────
2017                     ───
```

And compare different parties.

---

# 10. Party-level aggregation

This is an extremely important stage.

Your raw observations are **speeches**, but your research question is about **parties**.

Therefore aggregate speech-level features to party/time level.

For example:

```text
party × month
party × quarter
party × year
```

I recommend **month or quarter** for the main analysis because yearly aggregation can hide short-term responses to migration spikes.

For each party/month calculate:

* number of speeches
* average migration intensity
* average topic proportions
* average sentiment
* average embedding
* average speech length
* security-language share
* humanitarian-language share
* integration-language share

---

# 11. Modelling

This is where you answer the research question statistically.

I would divide modelling into several complementary approaches.

---

# 11.1 Model 1 — Topic prevalence over time

For topic \(k\):

$$
Topic_{p,t,k} =
\beta_0 +
\beta_1 Spike_t +
\beta_2 PostSpike_t +
\beta_3 Party_p +
\beta_4 Time_t +
\epsilon
$$

where:

* \(p\) = party
* \(t\) = month
* `Spike` = migration spike
* `PostSpike` = period following spike

This tells you whether topic prevalence changes around migration spikes.

---

# 11.2 Model 2 — Party-specific responses

The really interesting question is whether **different parties react differently**.

Therefore introduce an interaction:

$$
Topic_{p,t} =
\beta_0 +
\beta_1 Spike_t +
\beta_2 Party_p +
\beta_3(Spike_t \times Party_p)
+\epsilon
$$

The interaction tells you whether the relationship between migration spikes and speech changes differs by party.

---

# 11.3 Model 3 — Before/after analysis

Compare:

```text
Pre-spike
      ↓
Migration spike
      ↓
Post-spike
```

For each party measure:

$$
\Delta Topic =
Topic_{post} - Topic_{pre}
$$

This gives you a direct measure of **rhetorical change**.

---

# 11.4 Model 4 — Change in vocabulary

Calculate party vocabulary distributions before and after each spike.

Then measure distance.

Possible metrics:

### Cosine distance

$$
D_{cos}(A,B)=1-\frac{A\cdot B}{||A||||B||}
$$

### Jensen-Shannon divergence

Particularly useful for comparing probability distributions.

This lets you ask:

> How much did a party's language change following a migration spike?

---

# 11.5 Model 5 — Semantic change

Using embeddings:

$$
Embedding_{party,pre}
$$

versus

$$
Embedding_{party,post}
$$

Calculate semantic distance.

This gives you a more sophisticated measure of rhetorical change than simply counting words.

---

# 11.6 Model 6 — Party clustering

You can cluster parties based on their speech.

Possible inputs:

* TF-IDF
* topic proportions
* embeddings
* dictionary features

Methods:

* K-means
* hierarchical clustering
* DBSCAN

Then examine whether party groupings change after migration spikes.

---

# 12. Evaluation

Evaluation should happen at multiple levels.

## 12.1 Data quality evaluation

Report:

* missing values
* removed observations
* duplicate observations
* number of speeches before/after preprocessing
* number of parties
* number of countries
* temporal coverage

---

## 12.2 Topic model evaluation

Evaluate:

* coherence
* topic diversity
* topic stability
* human interpretability

Ideally, manually inspect the top words and representative speeches for each topic.

---

## 12.3 Classification evaluation

If you build a classifier predicting party/group from speech:

Use:

* accuracy
* precision
* recall
* F1
* confusion matrix

But remember that high classification performance isn't necessarily evidence that the party rhetoric is substantively different; it only means the model can distinguish the texts.

---

## 12.4 Robustness checks

This is particularly important for a political text-mining project.

Repeat the analysis using:

* different spike definitions
* different spike windows
* different topic numbers
* original vs translated text
* monthly vs quarterly aggregation
* TF-IDF vs embeddings
* alternative migration measures

If the main conclusions persist, your findings become more credible.

---

# 13. Visualizations

This project can have very strong visual outputs.

## 13.1 Migration timeline

Plot:

```text
Migration volume
        │
        │       /\ 
        │      /  \
        │_____/    \____
        └─────────────────
              Time
```

Mark identified migration spikes.

---

## 13.2 Speech volume over time

Plot the number of migration-related speeches per month/year.

This should be compared with actual migration levels.

---

# 13.3 Topic prevalence over time

For each major topic:

```text
x-axis = time
y-axis = topic prevalence
line = party
```

This is likely one of your most important figures.

---

# 13.4 Party × topic heatmap

Create:

|         | Border | Humanitarian | Integration | Economy |
| ------- | -----: | -----------: | ----------: | ------: |
| Party A |    .32 |          .15 |         .21 |     .08 |
| Party B |    .12 |          .34 |         .25 |     .17 |
| Party C |    .28 |          .18 |         .10 |     .21 |

Then produce the same heatmap for:

* pre-spike
* spike
* post-spike

---

# 13.5 Change heatmap

Perhaps even more informative:

| Party | Border | Humanitarian | Integration | Economy |
| ----- | -----: | -----------: | ----------: | ------: |
| A     |   +.12 |         -.05 |        +.03 |    -.01 |
| B     |   -.02 |         +.14 |        +.07 |    -.03 |
| C     |   +.20 |         -.11 |        -.04 |    +.02 |

This directly shows **rhetorical evolution**.

---

# 13.6 Word clouds

You can produce word clouds for:

* before spikes
* during spikes
* after spikes

and perhaps for individual parties.

But use these mainly as **illustrative visualizations**, not as your main evidence.

---

# 13.7 Semantic space

Using embeddings + dimensionality reduction:

* PCA
* t-SNE
* UMAP

Plot speeches in semantic space.

Color by:

* party
* time
* spike status

You could visually examine whether speech clusters shift following migration spikes.

---

# 13.8 Party trajectories

This could be one of your strongest visualizations.

Imagine each party moving through semantic space over time:

```text
            humanitarian
                  ↑
                  |
       Party A •────• 
              2014  2016
                  |
                  |
                  └────────────→ security
```

You can show the trajectory before → spike → after.

---

# 14. Conclusions

Your conclusion should answer **specific empirical questions**, rather than simply describing the NLP models.

Structure it around:

### 1. Attention

> Did parties talk more about migration during major migration spikes?

### 2. Topics

> Which migration-related topics became more prevalent?

### 3. Framing

> Did the balance between security, humanitarian, integration, economic and other frames change?

### 4. Party differences

> Did different parties respond differently?

### 5. Persistence

> Were changes temporary or did they persist after the spike?

### 6. Semantic change

> Did the overall language used by parties become substantially different?

### 7. Cross-national differences

> Were responses similar across countries or did national contexts matter?

---

# Recommended overall architecture

I would organize the actual project like this:

```text
01_data/
    raw/
    external_migration/
    processed/

02_notebooks/
    01_data_understanding.ipynb
    02_migration_spikes.ipynb
    03_text_preprocessing.ipynb
    04_exploratory_text_analysis.ipynb
    05_tfidf.ipynb
    06_topic_modeling.ipynb
    07_embeddings.ipynb
    08_party_analysis.ipynb
    09_statistical_models.ipynb
    10_evaluation.ipynb
    11_visualizations.ipynb

03_src/
    preprocessing.py
    features.py
    topics.py
    embeddings.py
    modelling.py
    visualization.py

04_results/
    figures/
    tables/
    models/

05_report/
    methodology/
    results/
    discussion/
```

---

# The complete pipeline in one diagram

```text
                    RAW DATA
                       │
                       ▼
             1. DATA UNDERSTANDING
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     Missing       Time/party      Text
      values        structure     quality
        │              │              │
        └──────────────┼──────────────┘
                       ▼
             2. MIGRATION DATA
                       │
                       ▼
              IDENTIFY SPIKES
                       │
                       ▼
                EVENT WINDOWS
                       │
                       ▼
             3. TEXT PREPROCESSING
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Clean       Tokenize      Lemmatize
          │            │            │
          └────────────┼────────────┘
                       ▼
            4. FEATURE EXTRACTION
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
     TF-IDF          Topics         Embeddings
       │               │                │
       └───────────────┼────────────────┘
                       ▼
             Dictionary features
                       │
                       ▼
                  5. MODELLING
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
    Time/topic      Party × spike    Semantic
     models           models          change
       │               │                │
       └───────────────┼────────────────┘
                       ▼
                  6. EVALUATION
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Topic        Statistical   Robustness
      quality        validity       checks
          │            │            │
          └────────────┼────────────┘
                       ▼
                7. VISUALIZATION
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
    Time series     Heatmaps       Party
                                    trajectories
                       │
                       ▼
                 8. CONCLUSIONS
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
    Attention       Framing       Party evolution
                       │
                       ▼
                 ANSWER RESEARCH
                    QUESTION
```

## My recommended "core" methodology

If this is a university project and you want a pipeline that is **ambitious but still manageable**, I would make these the core components:

1. **Clean and understand the speech corpus**
2. **Merge it with monthly migration statistics**
3. **Define migration spikes objectively**
4. **Create pre/during/post-spike windows**
5. **TF-IDF analysis**
6. **Dictionary-based migration/framing measures**
7. **LDA or another interpretable topic model**
8. **Aggregate results by party × month**
9. **Model party × migration-spike interactions**
10. **Use embeddings as a secondary semantic-change analysis**
11. **Perform robustness checks**
12. **Visualize party trajectories and topic changes**
13. **Interpret the results substantively**

The most important methodological point is that **the project should not become simply "topic modeling of EU speeches."** The central analytical structure should be **migration shock → party speech response → evolution over time**. That gives you a clear connection between the text-mining methods and the actual research question.
