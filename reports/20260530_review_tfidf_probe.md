# KMU RecSys 26 Steam — review TF-IDF validation probe

Review-enhanced/LLM recommendation papers suggest using text, but test pairs have no text. This probe uses only train-review user/item profiles and scores candidate cosine similarity. No Kaggle submission was performed.

| split | best score | row acc | per-user mean acc | vocab |
|---|---|---:|---:|---:|
| `val_random_sqrtpop_seed42` | `score_item_review_count` | 0.613923 | 0.629356 | 50000 |
| `val_recent_sqrtpop_seed42` | `score_item_review_count` | 0.586017 | 0.586347 | 50000 |
| `val_random_popbin_seed42` | `score_review_tfidf_user_item_cosine` | 0.533807 | 0.544136 | 50000 |
| `val_random_uniform_seed42` | `score_item_review_count` | 0.720044 | 0.741645 | 50000 |

## Full table

### val_random_sqrtpop_seed42

| score | row acc | per-user mean acc |
|---|---:|---:|
| `score_item_review_count` | 0.613923 | 0.629356 |
| `score_review_tfidf_user_item_cosine` | 0.579016 | 0.592723 |
| `score_user_review_count` | 0.498600 | 0.497638 |

### val_recent_sqrtpop_seed42

| score | row acc | per-user mean acc |
|---|---:|---:|
| `score_item_review_count` | 0.586017 | 0.586347 |
| `score_review_tfidf_user_item_cosine` | 0.585917 | 0.593029 |
| `score_user_review_count` | 0.498600 | 0.497638 |

### val_random_popbin_seed42

| score | row acc | per-user mean acc |
|---|---:|---:|
| `score_review_tfidf_user_item_cosine` | 0.533807 | 0.544136 |
| `score_item_review_count` | 0.522104 | 0.528786 |
| `score_user_review_count` | 0.498600 | 0.497638 |

### val_random_uniform_seed42

| score | row acc | per-user mean acc |
|---|---:|---:|
| `score_item_review_count` | 0.720044 | 0.741645 |
| `score_review_tfidf_user_item_cosine` | 0.632226 | 0.652042 |
| `score_user_review_count` | 0.498600 | 0.497638 |
