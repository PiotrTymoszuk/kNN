# This script provides functions to implement the k-NN algorithm
# for any distance measure provided by the package philentropy

  require(philentropy)

  knn <- function(train, test_vec, k, outcome,  method = 'jaccard', predict = T, ...){
    
    ## chooses the nearest neighbors or 
    ## predicts the feature (outcome) value for the given vector of features based on the k-nearest neighbors
    ## the solution modified from: 
    # https://stackoverflow.com/questions/50428127/implementing-knn-with-different-distance-metrics-using-r
    ## works for binary variables
    ## handles multiple data frame as a test vector by recursion
    
    if(any(class(test_vec) == 'data.frame') & nrow(test_vec) > 1) {
      
      pred_results <- 1:nrow(test_vec) %>% 
        map_dfr(function(x) knn(train = train, 
                                test_vec = test_vec[x, ], 
                                k = k, 
                                outcome = outcome, 
                                method = method, 
                                predict = predict, ...))
      
      pred_results <- pred_results %>% 
        mutate(test_set_id = rownames(test_vec))
      
      return(pred_results)
      
    }
    
    test_wo_outcome <- test_vec[, names(test_vec) != outcome]
    
    train_wo_outcome <- train[, names(train) != outcome]
    
    
    if(ncol(test_vec) != ncol(train)) {
      
      stop('the test and treining data sets must contain the same number of variables')
      
    }
    
    dist_storage <- 1:nrow(train) %>% 
      map_dfr(function(x) tibble(id_train = x, 
                                 distance = distance(rbind(test_wo_outcome, 
                                                           train_wo_outcome[x, ]), 
                                                     method = method, 
                                                     mute.message = T, ...)))
    
    dist_storage <- dist_storage %>% 
      mutate(outcome = train[[outcome]]) %>% 
      set_names(c('ID_train', 
                  'distance', 
                  outcome))
    
    neigh <- dist_storage %>% 
      top_n(n = k, 
            desc(distance))
    
    if(!predict) {
      
      return(neigh)
      
    } else {
      
      ## voting
      
      decision_stat <- mean(neigh[[outcome]])
      
      prediction <- ifelse(decision_stat == 0.5, 
                           sample(c(1, 0), 1), 
                           ifelse(decision_stat > 0.5, 1, 0))
      
      ## voting certainity obtained by Poisson test
      
      pred_certain <- poisson.test(c(sum(neigh[[outcome]]), 
                                     length(neigh[[outcome]]) - sum(neigh[[outcome]])))
      
      pred_tbl <- tibble(prediction = prediction, 
                         decision_stat = decision_stat, 
                         p_val = pred_certain$p.value) %>% 
        set_names(c(outcome, 
                    'voting_result', 
                    'p_value'))
      
      return(pred_tbl)
      
    }
    
  }
  
  check_accuracy <- function(train, test_vec, k, outcome, detailed = F) {
    
    ## compares the predictions with the real results
    ## returns either a summary table or detailed results
    
    prediction <- knn(train = train, 
                      test_vec = test_vec, 
                      k = k, 
                      outcome = outcome) %>% 
      set_names(c('pred_outcome', 
                  'voting_result', 
                  'p_value', 
                  'test_set_id'))
    
    real_outcome <- test_vec %>% 
      rownames_to_column('test_set_id') %>% 
      select(all_of(c('test_set_id', 
                      outcome))) %>% 
      set_names(c('test_set_id', 
                  'true_outcome'))
    
    comparison <- left_join(prediction, 
                            real_outcome, 
                            by = 'test_set_id') %>% 
      mutate(tp = (pred_outcome == 1 & true_outcome == 1), 
             tn = (pred_outcome == 0 & true_outcome == 0), 
             fp = (pred_outcome == 1 & true_outcome == 0), 
             fn = (pred_outcome == 0 & true_outcome == 1), 
             oob = fp|fn, 
             correct = tp|tn)
    
    summary <- tibble(Se = sum(comparison$tp)/sum(comparison$true_outcome), 
                      Sp = sum(comparison$tn)/(nrow(comparison) - sum(comparison$true_outcome)), 
                      error_rate = sum(comparison$oob)/nrow(comparison), 
                      correct_rate = sum(comparison$correct)/nrow(comparison))
    
    if(detailed) {
      
      return(pred_results = comparison, 
             stats = summary)
      
    } else {
      
      return(summary)
      
    }
    
  }