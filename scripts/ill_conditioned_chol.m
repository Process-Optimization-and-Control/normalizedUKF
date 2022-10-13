sig1 = 1e2;
sig2 = 1e-9;
delta = 1e-3;

std_dev = diag([sig1; sig2]);

corr = [1, delta;
    delta, 1];

P0 = std_dev*corr*std_dev; %covariance matrix

triangle = 'lower';
L_P0 = chol(P0, triangle);
L_corr = chol(corr, triangle);
L_P02 = std_dev*L_corr;

L_P0 == L_P02

%3x3 matrix
sig3 = 1e-8;
corr2 = [1, delta, 2*delta;
    delta, 1, sqrt(delta);
    2*delta, sqrt(delta), 1];

std_dev2 = diag([sig1, sig2, sig3]);
P2 = std_dev2*corr2*std_dev2;

L_P2 = chol(P2, triangle);
L_corr2 = chol(corr2, triangle);
L_P22 = std_dev2*L_corr2;

L_P22 == L_P2
