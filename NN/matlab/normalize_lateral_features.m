function y = normalize_lateral_features(x, normalizationJson)
% Clip and z-score exactly as the Python preprocessing pipeline.
stats = jsondecode(fileread(normalizationJson));
low = reshape(stats.clip_low, 1, []);
high = reshape(stats.clip_high, 1, []);
mu = reshape(stats.mean, 1, []);
sigma = reshape(stats.std, 1, []);
y = (min(max(double(x), low), high) - mu) ./ sigma;
end

