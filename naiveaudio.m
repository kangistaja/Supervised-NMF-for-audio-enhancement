
clear all

% Read the audio file
[audio, Fs] = audioread('174-168635-0012.flac');

% Define STFT parameters
fftLength = 512;
winLength = fftLength;
overlapLength = winLength / 2;

% Take the STFT of the audio signal with specific parameters
[audio_stft, F, T] = stft(audio, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);

% Design a low-pass filter using fir1
N = 100; % Order of the filter
cutoff = 0.3; % Normalized cutoff frequency (relative to Nyquist rate)
b = fir1(N, cutoff,'low');

figure;
plot(b)
title('Filter in time domain');

% Take FFT of the filter, ensure it's the correct length
filter_fft = fft(b, fftLength);

% Use fftshift to align the filter correctly with the STFT output
filter_fft = fftshift(filter_fft);

figure;
plot(abs(filter_fft))
title('Magnitude Response of the Filter');

% Apply the filter in the Fourier domain
filtered_audio_stft = audio_stft .* abs(filter_fft');

% Add noise in the Fourier domain
noise = randn(size(audio));
[noise_stft, F, T] = stft(noise, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength);
phase = angle(audio_stft);
filtered_noisy_audio_stft = filtered_audio_stft + 0.001*noise_stft;

% Convert to audio domain for listening using ISTFT with specific parameters
filtered_noisy_audio = real(istft(filtered_noisy_audio_stft, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength));

figure;
spectrogram(audio, winLength, overlapLength, fftLength, Fs, 'yaxis');
title('Original spectrogram');
xlabel('Time');
ylabel('Amplitude');

figure;
spectrogram(filtered_noisy_audio, winLength, overlapLength, fftLength, Fs, 'yaxis');
title('Filtered and noisy spectrogram');
xlabel('Time');
ylabel('Amplitude');

% Define a small noise level adjustment constant
mu = 0.01;

% Attempt naive deconvolution by
% dividing the Fourier transform of the noisy, filtered signal by the
% Fourier transform of the filter
recovered_audio_stft = filtered_noisy_audio_stft ./ (abs(filter_fft')+mu);

% Inverse FFT to get the time-domain signal using ISTFT with specific parameters
recovered_audio = real(istft(recovered_audio_stft, Fs, 'Window', hamming(winLength, 'periodic'), 'OverlapLength', overlapLength, 'FFTLength', fftLength));

% Listen to the audios
soundsc(audio, Fs);
pause(length(audio)/Fs + 2);
soundsc(filtered_noisy_audio, Fs);
pause(length(filtered_noisy_audio)/Fs + 2);
soundsc(recovered_audio, Fs);
pause(length(recovered_audio)/Fs + 2);

figure;
spectrogram(recovered_audio, winLength, overlapLength, fftLength, Fs, 'yaxis');
title('Deconvolved spectrogram');
xlabel('Time');
ylabel('Amplitude');

% Plot the original, filtered noisy, and recovered signals
t = (0:length(audio)-1)/Fs;
figure;
plot(t, audio);
title('Original Audio');
xlabel('Time (s)');
ylabel('Amplitude');

t = (0:length(filtered_noisy_audio)-1)/Fs;
figure;
plot(filtered_noisy_audio);
title('Filtered Noisy Audio');
xlabel('Time (s)');
ylabel('Amplitude');

t = (0:length(recovered_audio)-1)/Fs;
figure;
plot(recovered_audio);
title('Recovered Audio');
xlabel('Time (s)');
ylabel('Amplitude');

% Calculate SI-SDR for the recovered audio compared to the original
si_sdr_value = SI_SDR(audio, recovered_audio);
fprintf('SI-SDR of the recovered audio: %f dB\n', si_sdr_value);

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
    si_sdr = 10 * log10(dot(scaling_factor  * reference, scaling_factor * reference) / dot(error_signal, error_signal));
end