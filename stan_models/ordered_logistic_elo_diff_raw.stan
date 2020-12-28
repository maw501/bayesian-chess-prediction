data {
  int num_games;
  vector[num_games] p1_elo_diff_p2;
  int player_1_white[num_games];
  int<lower=1, upper=3> y[num_games];
  
  // Test set:
  int num_test_games;
  vector[num_test_games] p1_elo_diff_p2_test;
  int p1_test_white[num_test_games];

}
parameters {
  real e;
  real w;
  ordered[2] c;
}
model {
  real mu;
  vector[3] theta;

  for (i in 1:num_games) {
    mu = w*player_1_white[i] + e*p1_elo_diff_p2[i];
    
    theta[3] = inv_logit(mu - c[2]);  // prob win
    theta[2] = inv_logit(mu - c[1])  - inv_logit(mu - c[2]);  // prob draw
    theta[1] = 1 - inv_logit(mu - c[1]);  // prob lose
    y[i] ~ categorical(theta);
  }
}
generated quantities {
  vector[num_games] ypred;
  vector[num_test_games] ypred_test;
  real mu;
  vector[3] theta;
  
  for (i in 1:num_games) {
    mu = w*player_1_white[i] + e*p1_elo_diff_p2[i];
    
    theta[3] = inv_logit(mu - c[2]);  // prob win
    theta[2] = inv_logit(mu - c[1])  - inv_logit(mu - c[2]);  // prob draw
    theta[1] = 1 - inv_logit(mu - c[1]);  // prob lose
    ypred[i] = categorical_rng(theta);
  }
  
  for (i in 1:num_test_games) {
    mu = w*p1_test_white[i] + e*p1_elo_diff_p2_test[i];
    
    theta[3] = inv_logit(mu - c[2]);  // prob win
    theta[2] = inv_logit(mu - c[1])  - inv_logit(mu - c[2]);  // prob draw
    theta[1] = 1 - inv_logit(mu - c[1]);  // prob lose
    ypred_test[i] = categorical_rng(theta);
  }
}
