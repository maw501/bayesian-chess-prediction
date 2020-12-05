data {
  int num_players;
  int num_games;
  int<lower=2> K;
  vector[num_players] prior_score;
  int player_1_rank[num_games];
  int player_2_rank[num_games];
  int player_1_white[num_games];
  real b_mu;
  real<lower=0> b_scale;
  real sigma_a_mu;
  real<lower=0> sigma_a_scale;
  vector[K-1] c_mu;
  real<lower=0> c_scale;
  real w_mu;
  real<lower=0> w_scale;
}
parameters {
  real b;
  real<lower=0> sigma_a;
  ordered[K-1] c;
  vector[num_players] raw_a;
  real w;
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
  sigma_a ~ normal(sigma_a_mu, sigma_a_scale);
  w ~ normal(w_mu, w_scale);
}
generated quantities {
  vector[num_games] ypred;
  
  // Compute prior-predictive distribution for each game
  for (i in 1:num_games) {
    ypred[i] = ordered_logistic_rng(a[player_1_rank[i]] - a[player_2_rank[i]] + w*player_1_white[i], c);
  }
}