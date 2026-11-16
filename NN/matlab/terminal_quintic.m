function [t, y, yd, ydd, jerk] = terminal_quintic(terminalState, horizon, samples)
% Reconstruct a quintic with zero initial lateral state.
if nargin < 2, horizon = 2.5; end
if nargin < 3, samples = 51; end
A = [horizon^3 horizon^4 horizon^5; ...
     3*horizon^2 4*horizon^3 5*horizon^4; ...
     6*horizon 12*horizon^2 20*horizon^3];
a = A \ double(terminalState(1:3)).';
t = linspace(0, horizon, samples);
y = a(1)*t.^3 + a(2)*t.^4 + a(3)*t.^5;
yd = 3*a(1)*t.^2 + 4*a(2)*t.^3 + 5*a(3)*t.^4;
ydd = 6*a(1)*t + 12*a(2)*t.^2 + 20*a(3)*t.^3;
jerk = 6*a(1) + 24*a(2)*t + 60*a(3)*t.^2;
end

