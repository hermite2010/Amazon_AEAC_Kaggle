library(tidymodels)
library(embed)     # For target encoding
library(vroom) 
library(parallel)
#library(ggmosaic)  # For plotting
#library(ranger)    # FOR RANDOM/CLASSIFICATION FOREST
library(doParallel)
library(discrim) # FOR NAIVE BAYES
library(kknn)

# Reading in the Data
AEAC_Train <- vroom("train.csv") #"Amazon_AEAC_Kaggle/train.csv" for local
AEAC_Test <- vroom("test.csv") #"Amazon_AEAC_Kaggle/test.csv" for local

AEAC_Train$ACTION = as.factor(AEAC_Train$ACTION)

AEAC_recipe <- recipe(ACTION ~., data=AEAC_Train) %>% 
  step_mutate_at(all_numeric_predictors(), fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())
# Try step_lencode_bayes() in the future

prep <- prep(AEAC_recipe)
baked_data <- bake(prep, new_data=AEAC_Train)


# K-nearest neighbor ---------------------------------------------------

## Set up model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(knn_model)

## set up grid of tuning values

knn_tuning_grid <- grid_regular(neighbors(),
                               levels = 10)

## set up k-fold CV

knn_folds <- vfold_cv(AEAC_Train, v = 10, repeats=1)

## Set up Parallel processing

num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
if (num_cores > 4)
  num_cores = 10
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

## Run the CV

CV_results <- knn_wf %>%
  tune_grid(resamples=knn_folds,
            grid=knn_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

stopCluster(cl)
## find best tuning parameters

bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and prediction 

final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=AEAC_Train)

knn_preds <- final_wf %>%
  predict(new_data = AEAC_Test, type="prob") %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

vroom_write(x=knn_preds, file="KNN.csv", delim=",") #"Amazon_AEAC_Kaggle/KNN.csv"
