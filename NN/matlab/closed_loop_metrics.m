function metrics = closed_loop_metrics(logs, laneHalfWidth, sampleTime)
% Compute common HWA/LKA/ELK metrics from logged Simulink signals.
% Required fields: lateralOffset, lateralAcceleration, steeringAngle,
% pathLateral, fallbackActive, inferenceTime. pathLateral is time-by-points.
if nargin < 2, laneHalfWidth = 1.85; end
if nargin < 3, sampleTime = 0.05; end
offset = double(logs.lateralOffset(:));
acceleration = double(logs.lateralAcceleration(:));
steering = double(logs.steeringAngle(:));
departure = abs(offset) > laneHalfWidth;
metrics.laneDepartureCount = sum(diff([false; departure]) == 1);
metrics.laneDepartureFraction = mean(departure);
metrics.rmsLateralOffset = sqrt(mean(offset.^2));
metrics.maxLateralOffset = max(abs(offset));
metrics.maxLateralAcceleration = max(abs(acceleration));
metrics.maxLateralJerk = max(abs(diff(acceleration) / sampleTime));
metrics.maxSteeringRate = max(abs(diff(steering) / sampleTime));
path = double(logs.pathLateral);
metrics.pathJumpP95 = prctile(max(abs(diff(path, 1, 1)), [], 2), 95);
metrics.fallbackFraction = mean(logical(logs.fallbackActive(:)));
inference = double(logs.inferenceTime(:));
metrics.inferenceTimeMean = mean(inference);
metrics.inferenceTimeP95 = prctile(inference, 95);
outside = find(abs(offset) > 0.3, 1, 'last');
metrics.recoveryTime = 0;
if ~isempty(outside), metrics.recoveryTime = outside * sampleTime; end
end

