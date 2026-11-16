function terminal = lateral_tcn_forward(history, weights)
%LATERAL_TCN_FORWARD Exact eval-mode forward pass for the PyTorch TCN.
% history is [40 x 16], already clipped and normalized. terminal is [1 x 4].
x = double(history).'; % channels x time
channels = [32 48 64 64]; dilations = [1 2 4 8];
for block = 0:3
    prefix = sprintf('tcn_%d_', block);
    residual = x;
    residualWeight = [prefix 'residual_weight'];
    if isfield(weights, residualWeight)
        residual = pointwise(x, weights.(residualWeight), ...
            weights.([prefix 'residual_bias']));
    end
    y = causalConv(x, weights.([prefix 'conv1_weight']), ...
        weights.([prefix 'conv1_bias']), dilations(block+1));
    y = layerNormChannels(y, weights.([prefix 'norm1_norm_weight']), ...
        weights.([prefix 'norm1_norm_bias']));
    y = max(y, 0);
    y = causalConv(y, weights.([prefix 'conv2_weight']), ...
        weights.([prefix 'conv2_bias']), dilations(block+1));
    y = layerNormChannels(y, weights.([prefix 'norm2_norm_weight']), ...
        weights.([prefix 'norm2_norm_bias']));
    y = max(y, 0);
    x = max(y + residual, 0);
    assert(size(x,1) == channels(block+1));
end
z = x(:,end);
indices = [0 3 6 9];
for k = 1:numel(indices)
    prefix = sprintf('head_%d_', indices(k));
    headBias = weights.([prefix 'bias']);
    z = double(weights.([prefix 'weight'])) * z + double(headBias(:));
    if k < numel(indices), z = max(z, 0); end
end
terminal = z.';
end

function y = pointwise(x, weight, bias)
w = reshape(double(weight), size(weight,1), size(weight,2));
y = w * x + double(bias(:));
end

function y = causalConv(x, weight, bias, dilation)
w = double(weight); b = double(bias(:));
[outChannels, inChannels, kernel] = size(w);
steps = size(x,2); y = repmat(b, 1, steps);
for t = 1:steps
    for tap = 1:kernel
        source = t - dilation * (kernel - tap);
        if source >= 1
            y(:,t) = y(:,t) + reshape(w(:,:,tap), outChannels, inChannels) * x(:,source);
        end
    end
end
end

function y = layerNormChannels(x, gamma, beta)
mu = mean(x,1);
variance = mean((x-mu).^2,1);
y = (x-mu) ./ sqrt(variance + 1e-5);
y = y .* double(gamma(:)) + double(beta(:));
end
