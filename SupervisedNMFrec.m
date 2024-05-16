%Remember to run the supervisedNMF.m script before running this one!

% Read the audio file
[audio, Fs] = audioread('174-168635-0012.flac');

% Take the STFT of the audio signal
[audio_stft, F, T] = stft(audio, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);

% Define phase and magnitude
U = abs(audio_stft);
phi = angle(audio_stft);

% Design a low-pass filter using fir1
N = 100; % Order of the filter
cutoff = 0.3; % Normalized cutoff frequency (relative to Nyquist rate)
b = fir1(N, cutoff, 'low');

% Take FFT of the filter, ensure it's the correct length
filter_fft = fft(b, fftLength);

% Use fftshift to align the filter correctly with the STFT output
filter_fft = fftshift(filter_fft);

figure;
plot(abs(filter_fft))
title('Magnitude Response of the Filter');

% Apply the filter in the Fourier domain
filtered_audio_stft = audio_stft .* abs(filter_fft)';

% Add noise in the Fourier domain
noise = randn(size(audio));
[noise_stft, F, T] = stft(noise, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);
filtered_noisy_audio_stft = filtered_audio_stft + 0.00005 * noise_stft;

% Convert to audio domain for listening
filtered_noisy_audio = real(istft(filtered_noisy_audio_stft, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength));


% Define the exemplar initialized deconvolution dictionary

[num_rows, num_cols] = size(Utrain);

num_cols_W_dece = rank;  

rand_indices = randperm(num_cols, num_cols_W_dece);

W_dece = Utrain(:, rand_indices);

for i = 1:num_cols_W_dece
    W_dece(:, i) = W_dece(:, i) / norm(W_dece(:, i));
end

% Create a deconvolution dictionary by applying the filter to the dictionary
W_decr = W_rand .* abs(filter_fft)';
W_decn = W_nndsvd .* abs(filter_fft)';
W_dece = W_dece .* abs(filter_fft)';

% Initialize H randomly for the minimization process
H = rand(size(W_rand, 2), size(U, 2));

% Sparsity parameter for H
epsilon_H = 5; 
% Number of iterations for the minimization
max_iter = 500; 

U = abs(filtered_noisy_audio_stft);
phi = angle(filtered_noisy_audio_stft);

% Perform the minimization to find H_min for the deconvolution dictionaries using W_decr
for i = 1:max_iter
    H_numerator = W_decr' * U;
    H_denominator = W_decr' * W_decr * H + epsilon_H;
    H = H .* (H_numerator ./ H_denominator);
end
H_min_decr = H;

% Perform the minimization to find H_min for the deconvolution dictionaries using W_decn
H = rand(size(W_nndsvd, 2), size(U, 2));  % Reinitialize H for W_decn
for i = 1:max_iter
    H_numerator = W_decn' * U;
    H_denominator = W_decn' * W_decn * H + epsilon_H;
    H = H .* (H_numerator ./ H_denominator);
end
H_min_decn = H;

% Perform the minimization to find H_min for the deconvolution dictionaries using W_dece
H = rand(size(W_dece, 2), size(U, 2));  % Reinitialize H for W_dece
for i = 1:max_iter
    H_numerator = W_dece' * U;
    H_denominator = W_dece' * W_dece * H + epsilon_H;
    H = H .* (H_numerator ./ H_denominator);
end
H_min_dece = H;


%Define a noise level constant
mu = 0.01;

% Apply the Wiener-like filter using W_decr
recovered_audio_stft_decr = filtered_noisy_audio_stft .* (W_rand * H_min_decr) ./ (W_decr * H_min_decr + mu);

% Apply the Wiener-like filter using W_decn
recovered_audio_stft_decn = filtered_noisy_audio_stft .* (W_rand * H_min_decn) ./ (W_decn * H_min_decn + mu);

% Reconstruct audio
recovered_audio_decr = real(istft(recovered_audio_stft_decr, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength));
recovered_audio_decn = real(istft(recovered_audio_stft_decn, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength));
ft_decn = filtered_noisy_audio_stft .* (W_rand * H_min_decn) ./ (W_decn * H_min_decn + mu);

% Apply the Wiener-like filter using W_dece
recovered_audio_stft_dece = filtered_noisy_audio_stft .* (W_rand * H_min_dece) ./ (W_dece * H_min_dece + mu);

% Reconstruct audio
recovered_audio_decr = real(istft(recovered_audio_stft_decr, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength));
recovered_audio_decn = real(istft(recovered_audio_stft_decn, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength));
recovered_audio_dece = real(istft(recovered_audio_stft_dece, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength));

% Listen to the audios
soundsc(audio, Fs);
pause(length(audio)/Fs + 2);
soundsc(filtered_noisy_audio, Fs);
pause(length(filtered_noisy_audio)/Fs + 2);
soundsc(recovered_audio_decr, Fs);
pause(length(recovered_audio_decr)/Fs + 2);
soundsc(recovered_audio_decn, Fs);
pause(length(recovered_audio_decn)/Fs + 2);
soundsc(recovered_audio_dece, Fs);
pause(length(recovered_audio_dece)/Fs + 2);

% Plot the original, filtered noisy, and recovered signals
t = (0:length(audio)-1)/Fs;
figure;
plot(t, audio);
title('Original Audio');
xlabel('Time (s)');
ylabel('Amplitude');

figure;
plot(filtered_noisy_audio);
title('Filtered Noisy Audio');
xlabel('Time (s)');
ylabel('Amplitude');

figure;
plot(recovered_audio_decr);
title('Recovered Audio using W_decr');
xlabel('Time (s)');
ylabel('Amplitude');

figure;
plot(recovered_audio_decn);
title('Recovered Audio using W_decn');
xlabel('Time (s)');
ylabel('Amplitude');

figure;
plot(recovered_audio_dece);
title('Recovered Audio using W_dece');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the spectrograms of the original, filtered noisy, and recovered signals
figure;
spectrogram(audio, winLength, overlapLength, fftLength, Fs, 'yaxis');
title('Spectrogram of Original Audio');
xlabel('Time');
ylabel('Amplitude');

figure;
spectrogram(filtered_noisy_audio, winLength, overlapLength, fftLength, Fs, 'yaxis');
title('Spectrogram of Filtered Noisy Audio');
xlabel('Time');
ylabel('Amplitude');

figure;
spectrogram(recovered_audio_decr, winLength, overlapLength, fftLength, Fs, 'yaxis');
title('Deconvolved Spectrogram using W_decr');
xlabel('Time');
ylabel('Amplitude');

figure;
spectrogram(recovered_audio_decn, winLength, overlapLength, fftLength, Fs, 'yaxis');
title('Deconvolved Spectrogram using W_decn');
xlabel('Time');
ylabel('Amplitude');

figure;
spectrogram(recovered_audio_dece, winLength, overlapLength, fftLength, Fs, 'yaxis');
title('Deconvolved Spectrogram using W_dece');
xlabel('Time');
ylabel('Amplitude');

% Calculate SI-SDR for the recovered audio compared to the original using all dictionaries
si_sdr_decr = SI_SDR(audio, recovered_audio_decr);
fprintf('SI-SDR of the recovered audio using W_decr: %f dB\n', si_sdr_decr);

si_sdr_decn = SI_SDR(audio, recovered_audio_decn);
fprintf('SI-SDR of the recovered audio using W_decn: %f dB\n', si_sdr_decn);

si_sdr_dece = SI_SDR(audio, recovered_audio_dece);
fprintf('SI-SDR of the recovered audio using W_dece: %f dB\n', si_sdr_dece);

si_sdr_ovf = SI_SDR(audio, filtered_noisy_audio);
fprintf('SI-SDR of the filtered audio: %f dB\n', si_sdr_ovf);

% Calculate the SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
% Define the function for SI-SDR
function si_sdr = SI_SDR(reference, estimation)
    % Ensure the signals are aligned in time
    if length(reference) > length(estimation)
        reference = reference(1:length(estimation));
    else
        estimation = estimation(1:length(reference));
    end
    
    % Calculate the scale factor that minimizes the error
    scaling_factor = dot(estimation, reference) / dot(reference, reference);
    
    % Calculate the error signal
    error_signal = estimation - scaling_factor * reference;
    
    % Calculate SI-SDR
    si_sdr = 10 * log10(dot(scaling_factor * reference, scaling_factor * reference) / dot(error_signal, error_signal));
end