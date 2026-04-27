python train.py sweep --dataset-dir data/awgn_snr0_30/ --device cuda --sample-frac 0.25 --batch-size 512 --seed 123 --sweep-channel-formats real_imag mag_phase mag_inst_freq --epochs 10 --models time_cnn frequency_cnn spectrogram_cnn --sweep-cfo-estimators raw

python simulator.py generate --output-dir data/awgn_snr0_30 --snr-range 0,30 --n-signals 200000 --n-samples 2048 --sampler sobol
