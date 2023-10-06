# Amazon Employee Access Challenge AEAC

library(tidymodels)
library(embed) # For target encoding
library(vroom) 
library(ggmosaic) # For plotting

# Reading in the Data
AEAC_Train <- vroom("Amazon_AEAC_Kaggle/train.csv")
AEAC_Test <- vroom("Amazon_AEAC_Kaggle/test.csv")

AEAC_recipe <- recipe(ACTION ~., data=AEAC_Train) %>% 
  step_mutate_at(all_numeric_predictors(), fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors())# %>% 
#  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION))
  # Try step_lencode_bayes() in the future

prep <- prep(AEAC_recipe)
baked_data <- bake(prep, new_data=AEAC_Train)

length(baked_data)

ggplot(data = baked_data, aes(x = RESOURCE_other, fill = ACTION)) +
  geom_bar(position = "stack") +
  labs(title = "Stacked Bar Chart")+
  labs(
    x = "Other Resources",
    y = "Num of approved action",
    title = "Other Resources by ACTION")

ggplot(data = baked_data, aes(x = ROLE_DEPTNAME_other, fill = ACTION)) +
  geom_bar(position = "stack") +
  labs(title = "Stacked Bar Chart")+
  labs(
    x = "Other Department Names",
    y = "Num of approved action",
    title = "Other Departments by ACTION")
