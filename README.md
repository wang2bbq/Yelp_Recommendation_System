# Yelp_Recommendation_System
In this project, we are going to build a recommendation system using the Yelp datasets to predict the ratings/stars for given user ids and business ids. The datasets we are going to use are from: https://drive.google.com/drive/folders/1SIlY40owpVcGXJw3xeXk76afCwtSUx11?usp=sharing

# 1.Item-based CF recommendation system with Pearson similarity

# 2.Model-based recommendation system
- Method Description:
  - Use XGBregressor with advanced feature engineering(AutoEncoder) to predict ratings for each user and business pair.
    - 1.Pre-processing: train AutoEncoder model to learn continuous features from high-dimensional "categories" data. For assignment3 task2-2(last version), I included features like user's "review_count","average_stars","useful","funny","cool" and "fans", business's "stars","review_count" and "is_open". For this project(model_based), I've added "categories" for businesses, which is a string including multiple categories. After preprocessing and getting all the possible categories(1305), each category is transformed into a binary feature (0 or 1). Then I trained an AutoEncoder with a 16-neuron hidden layer to derive 16 significant new features. These embedded features are then incorporated into both the training and validation datasets.
    - 2.Hyperparameter tuning: use Optuna to tune hyperparameters of xgboost(using training set).
    - 3.Testing: combine the training set and validation set to train the final model and test on test set

# 3.Weighted Hybrid
- Combine them together using weighted average(adjust weights)
