if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

#----Data pre-processing------

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

######_______Training and Testing of Algorithm__________

# Create train and test sets
set.seed(1)
test_index <- createDataPartition(y=edx$rating, times = 1, p=0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Matches userId and movieId in both train and test sets
test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Adding back rows to the train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

#--------Exploratory analysis----------------------

# How many rows and columns are there in the edx dataset
str(edx)

# The summary function provides a statistical summary of the data.
edx %>% select(-genres) %>% summary()

# How many different movies, users are in the edx dataset
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

# The most popular movies in the dataset
edx %>%
  group_by(title) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  top_n(20, count) %>%
  ggplot(aes(count, reorder(title, count))) +
  geom_bar(color="black", fill = "turquoise", stat = "identity")+
  xlab("Count")+
  ylab(NULL)+
  theme_bw()

# The distribution of the movie ratings shows a range of 0.5 to 5 with whole numbers used more often.
edx %>%
  group_by(rating) %>%
  summarize(count = n())%>%
  ggplot(aes(rating,count))+
  geom_bar(color="black", fill = "deepskyblue", stat = "identity")+
  xlab("Ratings")+
  ylab("Relative Frequency")

# The distribution of numbers of users by numbers of ratings show right skew in the distribution
edx %>%
  group_by(userId) %>%
  summarise(count = n()) %>%
  ggplot(aes(count))+
  geom_histogram(color="black", fill = "turquoise", bins = 40)+
  xlab("Ratings")+
  ylab("Users")+
  scale_x_log10()+
  theme_bw()


#------Data Cleaning------
# Keep only the data import for the modeling to reduce processing time
train_set <- train_set %>% select(userId, movieId, rating, title)
test_set  <- test_set  %>% select(userId, movieId, rating, title)


######____________EVALUATION__________

# Function to calculate RMSE
RMSE <- function(actual_ratings, predicted_ratings){
  sqrt(mean((actual_ratings - predicted_ratings)^2))
}


######____________Model 1__________

# Prediction using only average of all movie ratings
mu_hat <- mean(train_set$rating)
mu_hat
mean_rmse <- RMSE(test_set$rating, mu_hat)
results <- tibble(Method = "Model 1: Simply Mean", RMSE = mean_rmse)
results %>% knitr::kable()

######____________Model 2__________

# Improve first model by taking into account movie bias i.e. popular movies receive higher ratings
bias_movie <- train_set %>%
  group_by(movieId) %>%
  summarize(bim = mean(rating-mu_hat))

# We can prove the bias by the reviewing distribution of ratings vs total
bias_movie %>% ggplot(aes(bim))+
  geom_histogram(color="black", fill = "turquoise", bins = 10)+
  xlab("Movie Bias")+
  ylab("Count")+
  theme_bw()

# Predicts using the test dataset
predict_ratings <- mu_hat + test_set %>%
  left_join(bias_movie, by="movieId") %>%
  pull(bim)

bias_movie_rmse <- RMSE(test_set$rating, predict_ratings)
results <- bind_rows(results, tibble(Method = "Model 2: Mean + movie bias", RMSE = bias_movie_rmse))
results %>% knitr::kable()

######____________Model 3__________

# Improves 1st model by taking into account user bias i.e. popular movies receive higher ratings
bias_user <- train_set %>%
  left_join(bias_movie, by = "movieId") %>%
  group_by(userId) %>%
  summarize(biu = mean(rating - mu_hat - bim))

# Predicts using the test dataset
predict_ratings <- test_set %>%
  left_join(bias_movie, by = "movieId") %>%
  left_join(bias_user, by = "userId") %>%
  mutate(pred = mu_hat+bim+biu) %>%
  pull(pred)

bias_movie_user_rmse <- RMSE(test_set$rating, predict_ratings)
results <- bind_rows(results, tibble(Method = "Model 3: Mean + Movie bias + User bias", RMSE =bias_movie_user_rmse))
results %>% knitr::kable()

######____________Model 4__________

# Some movies are rated by a few users and others by many adding variability and increasing RMSE. Requires regularization.

factors <- seq(0,8,1)
rmses <- sapply(factors, function(x){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+x))
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_hat)/(n()+x))
  predict_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu_hat + b_i + b_u) %>%
    pull(pred)
  return(RMSE(test_set$rating, predict_ratings))
})

# Plot that shows a range of lambdas VS RMSE. The optimal setting provides the lowest error.
qplot(factors, rmses)

results <- bind_rows(results, tibble(Method = "Model 4: Regularized movie & user effects", RMSE = min(rmses)))
results %>% knitr::kable()

######____________Model 5__________

#Recommender system for matrix factorization
library(recosystem)
set.seed(1)
train_rec <- with(train_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_rec <- with(test_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
rec_model <- Reco()

# Recosystem tuning
parameters <- rec_model$tune(train_rec, opts = list(dim = c(20,30),
                                                    costp_l2 = c(0.01, 0.1),
                                                    costq_l2 = c(0.01, 0.1),
                                                    lrate = c(0.01, 0.1),
                                                    nthread = 4,
                                                    niter = 10))

# Recosystem training
rec_model$train(train_rec, opts = c(parameters$min, nthread = 4, niter =30))

# Recosystem prediction
results_rec <- rec_model$predict(test_rec, out_memory())
factorization_rmse <- RMSE(test_set$rating, results_rec)

results <- bind_rows(results, tibble(Method = "Model 5: Matrix Factorization with Ricosystem", RMSE = factorization_rmse))
results %>% knitr::kable()


####__________FINAL VALIDATION_____
set.seed(1)
edx_reco <- with (edx, data_memory(user_index = userId, item_index = movieId, rating = rating))
validation_reco <- with(validation, data_memory(user_index = userId, item_index = movieId, rating = rating))
r <- Reco() 

para_reco <- r$tune(edx_reco, opts = list(dim = c(20,30),
                                            costp_l2 = c(0.01, 0.1),
                                            costq_l2 = c(0.01, 0.1),
                                            lrate = c(0.01, 0.1),
                                            nthread = 4,
                                            niter = 10))

r$train(edx_reco, opts = c(para_reco$min, nthread = 4, niter = 30))

# Final prediction
final_reco <- r$predict(validation_reco, out_memory())

# Final RMSE
final_rmse <- RMSE(validation$rating, final_reco)
results <- bind_rows(results, tibble(Method = "Final validation: Matrix factorization with Recosystem", RMSE = final_rmse))
results %>% knitr::kable()



