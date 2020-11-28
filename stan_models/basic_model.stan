data {
  int num_players;
  int num_games;
  vector[num_players] prior_score;
  int fave_rank[num_games];
  int underdog_rank[num_games];
  vector[num_games] fave_score;
  vector[num_games] underdog_score;
}
transformed data {
  vector[num_games] score_diff; 
  score_diff = fave_score - underdog_score;
}
parameters {
  real b;
  real<lower=0> sigma_a;
  real<lower=0> sigma_y;
  vector[num_players] raw_a;
}
transformed parameters {
  vector[num_players] a;
  
  a = b * prior_score + sigma_a * raw_a;
}
model {
  raw_a ~ std_normal();
  
  for (i in 1:num_games)
    score_diff[i] ~ normal(a[fave_rank[i]] - a[underdog_rank[i]], sigma_y);
}
generated quantities {
  vector[num_games] ypred;
  
  // Compute predictive distribution for each game
  for (i in 1:num_games) {
    ypred[i] = normal_rng(a[fave_rank[i]] - a[underdog_rank[i]], sigma_y);
  }
}