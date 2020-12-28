data {
  int num_games;
  vector[num_games] p1_elo_diff_p2;
  int player_1_white[num_games];
  int<lower=0, upper=1> fit_model;
  int<lower=1, upper=3> y[num_games];
  
  // Test set:
  int<lower=0, upper=1> predict_on_test_set;
  int num_test_games;
  vector[num_test_games] p1_elo_diff_p2_test;
  int p1_test_white[num_test_games];

}
parameters {
  real e;
  real w;
  real c_raw;
}
transformed parameters {
  ordered[2] c = to_vector({-c_raw, c_raw});
}
model {
  if (fit_model) {
    for (i in 1:num_games)
      y[i] ~ ordered_logistic(w*player_1_white[i] + e*p1_elo_diff_p2[i], c);
  }
}
generated quantities {
  vector[num_games] ypred;
  vector[num_test_games] ypred_test;

  // Compute prior/posterior predictive distribution for each game in training set:
  for (i in 1:num_games) {
    ypred[i] = ordered_logistic_rng(w*player_1_white[i] + e*p1_elo_diff_p2[i], c);
  }
  
  // Compute predictive distribution for test set:
  if (predict_on_test_set == 1) {
    for (i in 1:num_test_games) {
        ypred_test[i] = ordered_logistic_rng(w*p1_test_white[i] + e*p1_elo_diff_p2_test[i], c);
    }
  }
}