function maxError = validate_tcn_parity(weightsFile)
%VALIDATE_TCN_PARITY Compare dependency-free MATLAB inference with PyTorch.
data = load(weightsFile);
actual = lateral_tcn_forward(data.parity_input, data.weights);
expected = reshape(data.parity_expected,1,[]);
maxError = max(abs(actual-expected),[],'all');
fprintf('Maximum Python/MATLAB TCN error: %.12g\n', maxError);
assert(maxError < 1e-5, 'TCN inference parity exceeds 1e-5');
end
