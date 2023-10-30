library(tidymodels)
library(embed)     # For target encoding
library(vroom) 
library(parallel)
library(ggmosaic)  # For plotting
library(ranger)    # FOR RANDOM/CLASSIFICATION FOREST
library(doParallel)
library(discrim) # FOR NAIVE BAYES
library(kknn)
library(themis) # for smote

# Reading in the Data
AEAC_Train <- vroom("train.csv") #"Amazon_AEAC_Kaggle/train.csv" for local
AEAC_Test <- vroom("test.csv") #"Amazon_AEAC_Kaggle/test.csv" for local

AEAC_Train$ACTION = as.factor(AEAC_Train$ACTION)

AEAC_recipe <- recipe(ACTION ~., data=AEAC_Train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION)) %>% 
  step_smote(all_outcomes(), neighbors = 5) #%>% 
#  step_upsample()
# OR step_downsample()

# apply the recipe to your data
prepped_recipe <- prep(AEAC_recipe)
baked_data <- bake(prepped_recipe, new_data=AEAC_Train)


# Classification Forest ---------------------------------------------------
# Set Up the Engine
class_for_mod <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

## Workflow and model and recipe
class_for_wf <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(class_for_mod)

## set up grid of tuning values
class_tuning_grid <- grid_regular(mtry(range = c(1,(ncol(AEAC_Train)-1))),
                                  min_n(),
                                  levels = 6)

## set up k-fold CV
class_folds <- vfold_cv(AEAC_Train, v = 4, repeats=1)

# ## Set up Parallel processing
# num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
# if (num_cores > 4)
#   num_cores = 10
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)

## Run the CV
CV_results <- class_for_wf %>%
  tune_grid(resamples=class_folds,
            grid=class_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

#stopCluster(cl)
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

## Write it out
vroom_write(x=class_for_preds, file="ClassForest_SMOTE.csv", delim=",")

# Logistic Regression -----------------------------------------------------
#Type of model
log_reg_mod <- logistic_reg() %>% 
  set_engine("glm")

# Set Up Workflow
amazon_log_workflow <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(log_reg_mod) %>%
  fit(data = AEAC_Train) # Fit the workflow

amazon_log_predictions <- predict(amazon_log_workflow,
                                  new_data=AEAC_Test,
                                  type= "prob") #%>% # "class" which uses cutoff as .5 or "prob" (see doc)
#mutate(ACTION = ifelse(.pred_1> C,1,0)) #REPLACE C WITH WHATEVER CUTOFF I WANT

### FOR TYPE CLASS
# log_submission <- amazon_log_predictions %>% 
#   mutate(id = row_number()) %>% 
#   rename(Action = .pred_class)

### FOR TYPE PROB
log_submission <- amazon_log_predictions %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

## Write it out
vroom_write(x=log_submission, file="LogRegression_SMOTE.csv", delim=",")

# Penalized Logistic Regression -------------------------------------------
#Type of model
pen_log_model <- logistic_reg(mixture=tune(),
                              penalty=tune()) %>% 
  set_engine("glmnet")
# Set the Workflow
pen_log_workflow <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(pen_log_model)

## Grid of values to tune over
pen_log_tuning_grid <- grid_regular(penalty(),
                                    mixture(), # Always bewteen 0 and 1
                                    levels = 3) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(AEAC_Train, v = 3, repeats=1)

## Run the CV
CV_results <- pen_log_workflow %>%
  tune_grid(resamples=folds,
            grid=pen_log_tuning_grid,
            metrics=metric_set(roc_auc)) 

## Find Best Tuning Parameters
pen_log_bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- pen_log_workflow %>%
  finalize_workflow(pen_log_bestTune) %>%
  fit(data=AEAC_Train)

## Predict
pen_log_preds <- final_wf %>%
  predict(new_data = AEAC_Test, type="prob") %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

## Write it out
vroom_write(x=pen_log_preds, file="PenLogRegression_SMOTE.csv", delim=",")

# Naive Bayes -------------------------------------------------------------
## Set Up Model
nb_model <- naive_Bayes(Laplace=tune(), 
                        smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

## Workflow and model and recipe
nb_wf <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(nb_model)

## set up grid of tuning values
nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 10)

## set up k-fold CV
nb_folds <- vfold_cv(AEAC_Train, v = 5, repeats=1)

# ## Set up Parallel processing
# num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
# if (num_cores > 4)
#   num_cores = 10
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=nb_folds,
            grid=nb_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL
#stopCluster(cl)

## find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and prediction 
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=AEAC_Train)

nb_preds <- final_wf %>%
  predict(new_data = AEAC_Test, type="prob") %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

## Write it out
vroom_write(x=nb_preds, file="NaiveBayes_SMOTE.csv", delim=",")

# K Nearest Neighbor ------------------------------------------------------
## Set up model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

## Set the Workflow
knn_wf <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(knn_model)

## set up grid of tuning values
knn_tuning_grid <- grid_regular(neighbors(),
                                levels = 10)

## set up k-fold CV
knn_folds <- vfold_cv(AEAC_Train, v = 10, repeats=1)

## Set up Parallel processing
# num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
# if (num_cores > 4)
#   num_cores = 10
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=knn_folds,
            grid=knn_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

#stopCluster(cl)
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

## Write it out
vroom_write(x=knn_preds, file="KNN_SMOTE.csv", delim=",") #"Amazon_AEAC_Kaggle/KNN.csv"


# SVM Models ---------------------------------------------------------------------
## SVM Engines & Workflows
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
                             levels = 5)
svmRadial_grid <- grid_regular(rbf_sigma(),
                               cost(),
                               levels = 5)
svmLinear_grid <- grid_regular(cost(),
                               levels = 5)

## set up k-fold CV
svm_folds <- vfold_cv(AEAC_Train, v = 3, repeats=1)

## Set up Parallel processing
# num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
# if (num_cores > 4)
#   num_cores = 10
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)

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

#stopCluster(cl)

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

vroom_write(x=svmPoly_preds, file="SVM_Poly_SMOTE.csv", delim=",") #"Amazon_AEAC_Kaggle/SVM_Poly.csv"

svmRadial_preds <- radialFinal_wf %>%
  predict(new_data = AEAC_Test, type="prob") %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

vroom_write(x=svmRadial_preds, file="SVM_Radial_SMOTE.csv", delim=",")

svmLinear_preds <- linearFinal_wf %>%
  predict(new_data = AEAC_Test, type="prob") %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

vroom_write(x=svmPoly_preds, file="SVM_Linear_SMOTE.csv", delim=",")

