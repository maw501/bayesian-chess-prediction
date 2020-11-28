data {
  int num_players;
  int num_games;
  int<lower=2> K;
  vector[num_players] prior_score;
  int fave_rank[num_games];
  int underdog_rank[num_games];
  int<lower=1, upper=K> y[num_games];
}
parameters {
  real b;
  real<lower=0> sigma_a;
  //real<lower=0> sigma_y;
  vector[num_players] raw_a;
  ordered[K-1] c;
}
transformed parameters {
  vector[num_players] a;
  
  a = b * prior_score + sigma_a * raw_a;
}
model {
  raw_a ~ std_normal();
  
  for (i in 1:num_games)
    //score_diff[i] ~ normal(a[fave_rank[i]] - a[underdog_rank[i]], sigma_y);
    y[i] ~ ordered_logistic(a[fave_rank[i]] - a[underdog_rank[i]], c);
}
generated quantities {
  vector[num_games] ypred;
  
  // Compute predictive distribution for each game
  for (i in 1:num_games) {
    ypred[i] = ordered_logistic_rng(a[fave_rank[i]] - a[underdog_rank[i]], c);
  }
}