% Define STFT parameters
fftLength = 512;
winLength = fftLength;
overlapLength = winLength / 2;

% Read each audio file
[audio1, fs] = audioread('174-168635-0001.flac');
[audio2, fs] = audioread('174-168635-0002.flac');
[audio3, fs] = audioread('174-168635-0003.flac');
[audio4, fs] = audioread('174-168635-0004.flac');
[audio5, fs] = audioread('174-168635-0005.flac');
[audio6, fs] = audioread('174-168635-0006.flac');
[audio7, fs] = audioread('174-168635-0007.flac');
[audio8, fs] = audioread('174-168635-0008.flac');
[audio9, fs] = audioread('174-168635-0009.flac');
[audio10, fs] = audioread('174-168635-0010.flac');


% Compute the STFT for each audio file
[S1, ~, ~] = stft(audio1, fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);
[S2, ~, ~] = stft(audio2, fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);
[S3, ~, ~] = stft(audio3, fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);
[S4, ~, ~] = stft(audio4, fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);
[S5, ~, ~] = stft(audio5, fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);
[S6, ~, ~] = stft(audio6, fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);
[S7, ~, ~] = stft(audio7, fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);
[S8, ~, ~] = stft(audio8, fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);
[S9, ~, ~] = stft(audio9, fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);
[S10, ~, ~] = stft(audio10, fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);

% Concatenate the magnitude of the STFTs to form Utrain
Utrain = [abs(S1), abs(S2), abs(S3), abs(S4), abs(S5), abs(S6), abs(S7), abs(S8), abs(S9), abs(S10)];


% Now Utrain contains the concatenated STFTs of all the audio files
disp('Training matrix Utrain is ready.');


% Import MATLAB matrix operations library
import matlab.*

% define rank
rank = 64;

% Test random_initialization
[W_rand, H_rand] = random_initialization(Utrain, rank);
fprintf('Random Initialization:\n');
disp('W:');
disp(W_rand);
disp('H:');
disp(H_rand);

% Test nndsvd_initialization
[W_nndsvd, H_nndsvd] = nndsvd_initialization(Utrain, rank);
fprintf('NNDSVD Initialization:\n');
disp('W:');
disp(W_nndsvd);
disp('H:');
disp(H_nndsvd);

% Test multiplicative_update
max_iter = 500;
[W_mu, H_mu, norms] = multiplicative_update(Utrain, rank,max_iter, 'random');
fprintf('Multiplicative Update:\n');
disp('W:');
disp(W_mu);
disp('H:');
disp(H_mu);
fprintf('Frobenius Norms:\n');
disp(norms);

% Define random initialization function
function [W, H] = random_initialization(Utrain, rank)
    % Initialize matrices W and H randomly.

    % Parameters:
    % - Utrain: Input matrix
    % - rank: Rank of the factorization

    % Returns:
    % - W: Initialized W matrix
    % - H: Initialized H matrix

    num_docs = size(Utrain, 1);
    num_terms = size(Utrain, 2);
    W = 1 + (2-1).*rand(num_docs, rank);
    H = 1 + (2-1).*rand(rank, num_terms);
end

% Define NNDSVD initialization function
function [W, H] = nndsvd_initialization(Utrain, rank)
    % Initialize matrices W and H using Non-negative Double Singular Value Decomposition (NNDSVD).

    % Parameters:
    % - Utrain: Input matrix
    % - rank: Rank of the factorization

    % Returns:
    % - W: Initialized W matrix
    % - H: Initialized H matrix

    [U, S, V] = svd(Utrain, 'econ');
    W = zeros(size(Utrain, 1), rank);
    H = zeros(rank, size(Utrain, 2));

    W(:, 1) = sqrt(S(1,1)) * abs(U(:, 1));
    H(1, :) = sqrt(S(1,1)) * abs(V(:, 1))';

    for i = 2:rank
        ui = U(:, i);
        vi = V(:, i);
        ui_pos = (ui >= 0) .* ui;
        ui_neg = (ui < 0) .* -ui;
        vi_pos = (vi >= 0) .* vi;
        vi_neg = (vi < 0) .* -vi;

        ui_pos_norm = norm(ui_pos, 2);
        ui_neg_norm = norm(ui_neg, 2);
        vi_pos_norm = norm(vi_pos, 2);
        vi_neg_norm = norm(vi_neg, 2);

        norm_pos = ui_pos_norm * vi_pos_norm;
        norm_neg = ui_neg_norm * vi_neg_norm;

        if norm_pos >= norm_neg
            W(:, i) = sqrt(S(i,i) * norm_pos) / ui_pos_norm * ui_pos;
            H(i, :) = sqrt(S(i,i) * norm_pos) / vi_pos_norm * vi_pos';
        else
            W(:, i) = sqrt(S(i,i) * norm_neg) / ui_neg_norm * ui_neg;
            H(i, :) = sqrt(S(i,i) * norm_neg) / vi_neg_norm * vi_neg';
        end
    end
end

% Define multiplicative update function
function [W, H, norms] = multiplicative_update(Utrain, k, max_iter, init_mode)
    % Perform Multiplicative Update (MU) algorithm for Non-negative Matrix Factorization (NMF).

    % Parameters:
    % - Utrain: Input matrix
    % - k: Rank of the factorization
    % - max_iter: Maximum number of iterations
    % - init_mode: Initialization mode ('random' or 'nndsvd')

    % Returns:
    % - W: Factorized matrix W
    % - H: Factorized matrix H
    % - norms: List of Frobenius norms at each iteration

    if strcmp(init_mode, 'random')
        [W, H] = random_initialization(Utrain, k);
    elseif strcmp(init_mode, 'nndsvd')
        [W, H] = nndsvd_initialization(Utrain, k);
    end

    norms = zeros(1, max_iter);
    epsilon_H = 5;
    epsilon_W = 1.0e-10;

    for i = 1:max_iter
        % Update H
        W_TUtrain = W' * Utrain;
        W_TWH = W' * W * H + epsilon_H;
        H = H .* (W_TUtrain ./ W_TWH);

        % Update W
        UtrainH_T = Utrain * H';
        WHH_T = W * H * H' + epsilon_W;
        W = W .* (UtrainH_T ./ WHH_T);

        % Normalize W column-wise
        W_norms = sqrt(sum(W.^2, 1)) + 1e-10; % Adding a small constant to avoid division by zero
        W = W ./ W_norms;

        % Normalize H row-wise
        H = H .* W_norms';

        % Calculate the Frobenius norm of the difference between Utrain and WH
        norm_val = norm(Utrain - W * H, 'fro');
        norms(i) = norm_val;
    end
end