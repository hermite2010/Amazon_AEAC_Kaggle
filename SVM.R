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


# SVM ---------------------------------------------------

## SVM models
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmPoly_wf <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(svmPoly)

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmRadial_wf <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(svmRadial)

svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmLinear_wf <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(svmLinear)

## set up grid of tuning values
svmPoly_grid <- grid_regular(degree(),
                             cost(),
                             levels = 3)
svmRadial_grid <- grid_regular(rbf_sigma(),
                               cost(),
                               levels = 3)
svmLinear_grid <- grid_regular(cost(),
                               levels = 3)

## set up k-fold CV
svm_folds <- vfold_cv(AEAC_Train, v = 5, repeats=1)

## Set up Parallel processing
num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
if (num_cores > 4)
  num_cores = 10
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

## Run the CV
svmPoly_results <- svmPoly_wf %>%
  tune_grid(resamples=svm_folds,
            grid=svmPoly_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

svmRadial_results <- svmRadial_wf %>%
  tune_grid(resamples=svm_folds,
            grid=svmRadial_grid,
            metrics=metric_set(roc_auc))
            
svmLinear_results <- svmLinear_wf %>%
  tune_grid(resamples=svm_folds,
            grid=svmLinear_grid,
            metrics=metric_set(roc_auc))
            
stopCluster(cl)

## find best tuning parameters
polyBestTune <- svmPoly_results %>%
  select_best("roc_auc")

radialBestTune <- svmRadial_results %>%
  select_best("roc_auc")

linearBestTune <- svmLinear_results %>%
  select_best("roc_auc")

## Finalize workflow and prediction 
polyFinal_wf <- svmPoly_wf %>%
  finalize_workflow(polyBestTune) %>%
  fit(data=AEAC_Train)

radialFinal_wf <- svmRadial_wf %>%
  finalize_workflow(radialBestTune) %>%
  fit(data=AEAC_Train)

linearFinal_wf <- svmLinear_wf %>%
  finalize_workflow(linearBestTune) %>%
  fit(data=AEAC_Train)

svmPoly_preds <- polyFinal_wf %>%
  predict(new_data = AEAC_Test, type="prob") %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

vroom_write(x=svmPoly_preds, file="SVM_Poly.csv", delim=",") #"Amazon_AEAC_Kaggle/SVM_Poly.csv"

svmRadial_preds <- radialFinal_wf %>%
  predict(new_data = AEAC_Test, type="prob") %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

vroom_write(x=svmRadial_preds, file="SVM_Radial.csv", delim=",")

svmLinear_preds <- linearFinal_wf %>%
  predict(new_data = AEAC_Test, type="prob") %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

vroom_write(x=svmPoly_preds, file="SVM_Linear.csv", delim=",")
