data {
  int num_players;
  int num_games;
  int<lower=2> K;
  vector[num_players] prior_score;
  int<lower=1> player_1_rank[num_games];
  int<lower=1> player_2_rank[num_games];
  int player_1_white[num_games];
  real b_mu;
  real<lower=0> b_scale;
  real sigma_a_mu;
  real<lower=0> sigma_a_scale;
  vector[K-1] c_mu;
  real<lower=0> c_scale;
  real w_mu;
  real<lower=0> w_scale;
  int<lower=0, upper=1> fit_model;
  int<lower=1, upper=K> y[num_games];
  
  // Both players in the test set:
  int num_test_both_games;
  int<lower=1> p1_test_both_rank[num_test_both_games];
  int<lower=1> p2_test_both_rank[num_test_both_games];
  int p1_test_both_white[num_test_both_games];
  
  // Neither players in the test set:
  int num_test_neither_games;
  vector[num_test_neither_games] p1_test_neither_prior_score;
  vector[num_test_neither_games] p2_test_neither_prior_score;
  int p1_test_neither_white[num_test_neither_games];
  
  // Only one player in test set:
  int num_test_one_games;
  int<lower=0, upper=1> p1_in_train[num_test_one_games];
  int<lower=0, upper=1> p2_in_train[num_test_one_games];
  int<lower=1> p1_test_one_rank[num_test_one_games];
  int<lower=1> p2_test_one_rank[num_test_one_games];
  vector[num_test_one_games] p1_test_one_prior_score;
  vector[num_test_one_games] p2_test_one_prior_score;
  int p1_test_one_white[num_test_one_games];

}
parameters {
  real b;
  real w;
  ordered[K-1] c;
  real<lower=0> sigma_a;
  vector[num_players] raw_a;
}
transformed parameters {
  vector[num_players] a;
  
  a = b * prior_score + sigma_a * raw_a;
}
model {
  // priors
  raw_a ~ std_normal();
  b ~ normal(b_mu, b_scale);
  c ~ normal(c_mu, c_scale);
  w ~ normal(w_mu, w_scale);
  sigma_a ~ normal(sigma_a_mu, sigma_a_scale);
  
  // model
  if (fit_model) {
    for (i in 1:num_games)
      y[i] ~ ordered_logistic(a[player_1_rank[i]] - a[player_2_rank[i]] + w*player_1_white[i], c);
  }
}
generated quantities {
  vector[num_games] ypred;
  vector[num_test_both_games] ypred_test_both;
  vector[num_test_neither_games] ypred_test_neither;
  vector[num_test_one_games] ypred_test_one;
  
  // Compute prior/posterior predictive distribution for each game
  for (i in 1:num_games) {
    ypred[i] = ordered_logistic_rng(a[player_1_rank[i]] - a[player_2_rank[i]] + w*player_1_white[i], c);
  }
  
  // Compute predictive distribution when both players are in training set
  for (i in 1:num_test_both_games) {
    ypred_test_both[i] = ordered_logistic_rng(a[p1_test_both_rank[i]] - a[p2_test_both_rank[i]] + w*p1_test_both_white[i], c);
  }
  
  // Compute predictive distribution when neither players are in training set
  for (i in 1:num_test_neither_games) {
    ypred_test_neither[i] = ordered_logistic_rng(b * (p1_test_neither_prior_score[i] - p2_test_neither_prior_score[i]) + 
                                                 w*p1_test_neither_white[i], 
                                                 c
                                                 );
  }
  
  // Compute predictive distribution when only one player is training set
  for (i in 1:num_test_one_games) {
    if (p1_in_train[i] == 1) {
      ypred_test_one[i] = ordered_logistic_rng(a[p1_test_one_rank[i]] - b*p2_test_one_prior_score[i] + 
                                                 w*p1_test_one_white[i], 
                                                 c
                                                 );
    } else if (p2_in_train[i] == 1) {
      ypred_test_one[i] = ordered_logistic_rng(-a[p2_test_one_rank[i]] + b*p1_test_one_prior_score[i] + 
                                                 w*p1_test_one_white[i], 
                                                 c
                                                 );
    }
  }
}