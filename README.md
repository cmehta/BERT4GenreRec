# Introduction

In this research we are trying to improve the Movie recommendation system by predicting the next genre a user would prefer to watch. Often the star rating system used for providing these recommendations suffer from several shortcomings. Ratings are prone to grade inflation where the difference between 3.5 and 4 stars could be massive whereas 3 and 3.5 can be considered as equally bad. Also, lack of incentives for providing useful feedback result in biased opinions where only people with extreme experience share their feedback. However, information provided by experts and directors like genre can be considered very reliable as compared to user ratings.

To achieve the same we are implementing an end to end model inspired by:
> **BERT4Rec: Sequential Recommendation with BERT (Sun et al.)**  

and lets you train them on MovieLens-1m and MovieLens-20m.
# Data
## MovieLens-1 million
http://files.grouplens.org/datasets/movielens/ml-1m.zip

## MovieLens-20 million
http://files.grouplens.org/datasets/movielens/ml-20m.zip

Running the code would automatically download the required data, if it's not present.

# Usage

## Overall

Run `main.py` with arguments to train and/or test you model. There are predefined templates for all models.

On running `main.py`, it asks you whether to train on MovieLens-1m or MovieLens-20m. (Enter 1 or 20)

After training, it also asks you whether to run test set evaluation on the trained model. (Enter y or n)

## BERT4Rec

```bash
python main.py --template train_bert
```

## Examples

1. Train BERT4Rec on ML-20m and run test set inference after training

   ```bash
   printf '20\ny\n' | python main.py --template train_bert
   ```

# Test Set Results
## MovieLens-1m
| dataset       | Metrics | Test Result | Training Results| 
| ------------- |:-------:|:-----------:|:---------------:|
|               | HR@1    | 0.631       | 0.683           |
|               | HR@5    | 0.854       | 0.884           |
|               | HR@10   | 0.923       | 0.943           |
|               | HR@20   | 0.972       | 0.981           |
|               | HR@40   | 0.994       | 0.996           |
| Movie Lens 1M |:-------:|:-----------:|:---------------:|
|               | NDCG@1  | 0.631       | 0.683           |
|               | NDCG@5  | 0.75        | 0.791           |
|               | NDCG@10 | 0.772       | 0.810           |
|               | NDCG@20 | 0.785       | 0.820           |
|               | NDCG@40 | 0.79        | 0.823           |
|-------------  |:-------:|:-----------:|:---------------:|


# EDA
EDA of the dataset can be visualized under notebook folder
