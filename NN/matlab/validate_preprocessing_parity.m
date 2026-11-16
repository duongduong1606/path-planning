function maxError = validate_preprocessing_parity(fixtureDirectory)
input = readmatrix(fullfile(fixtureDirectory, 'input.csv'));
expected = readmatrix(fullfile(fixtureDirectory, 'expected.csv'));
actual = normalize_lateral_features(input, fullfile(fixtureDirectory, 'normalization.json'));
maxError = max(abs(actual - expected), [], 'all');
fprintf('Maximum Python/MATLAB preprocessing error: %.12g\n', maxError);
assert(maxError < 1e-5, 'Preprocessing parity exceeds 1e-5');
end

