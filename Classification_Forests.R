library(tidymodels)
library(embed)     # For target encoding
library(vroom) 
library(parallel)
library(ggmosaic)  # For plotting
library(ranger)    # FOR RANDOM/CLASSIFICATION FOREST
library(doParallel)
library(themis)

# Reading in the Data
AEAC_Train <- vroom("train.csv") #"Amazon_AEAC_Kaggle/train.csv" for local
AEAC_Test <- vroom("test.csv") #"Amazon_AEAC_Kaggle/test.csv" for local

AEAC_Train$ACTION = as.factor(AEAC_Train$ACTION)

AEAC_recipe <- recipe(ACTION ~., data=AEAC_Train) %>% 
  step_mutate_at(all_numeric_predictors(), fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>%  
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION)) %>%
  step_smote(all_outcomes(), neighbors = 5) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.95)

prep <- prep(AEAC_recipe)
baked_data <- bake(prep, new_data=AEAC_Train)


# Classification Forest ---------------------------------------------------

# Set Up the Engine

class_for_mod <- rand_forest(mtry = tune(),
                            min_n=tune(),
                            trees=750) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

## Workflow and model and recipe

class_for_wf <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(class_for_mod)

## set up grid of tuning values

class_tuning_grid <- grid_regular(mtry(range = c(1,(ncol(AEAC_Train)-1))),
                                 min_n(),
                                 levels = 10)

## set up k-fold CV

class_folds <- vfold_cv(AEAC_Train, v = 5, repeats=1)

## Set up Parallel processing

num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
if (num_cores > 4)
  num_cores = 10
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

## Run the CV

CV_results <- class_for_wf %>%
  tune_grid(resamples=class_folds,
            grid=class_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

stopCluster(cl)
## find best tuning parameters

bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and prediction 

final_wf <- class_for_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=AEAC_Train)

class_for_preds <- final_wf %>%
  predict(new_data = AEAC_Test, type="prob") %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

vroom_write(x=class_for_preds, file="ClassForest_smote+pca.csv", delim=",") #"Amazon_AEAC_Kaggle/PenLogRegression.csv"

