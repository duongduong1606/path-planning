function scenarios = tcn_scenario_contract()
%TCN_SCENARIO_CONTRACT Model-only closed-loop scenarios (no safety shield).
base = struct('duration',12,'speed',22,'road','straight', ...
    'initialOffset',0,'initialHeadingDeg',0, ...
    'disturbanceStart',inf,'disturbanceEnd',-inf, ...
    'lateralDisturbance',0,'yawDisturbance',0, ...
    'lowConfidenceStart',inf,'lowConfidenceEnd',-inf);
scenarios = repmat(base,6,1);
scenarios(1).name = 'HWA_nominal_straight';
scenarios(2).name = 'HWA_nominal_curve'; scenarios(2).road = 'curve';
scenarios(3).name = 'LKA_offset_recovery'; scenarios(3).road = 'curve';
scenarios(3).initialOffset = 0.5;
scenarios(4).name = 'LKA_heading_recovery'; scenarios(4).road = 'curve';
scenarios(4).initialHeadingDeg = 3;
scenarios(5).name = 'ELK_forced_departure'; scenarios(5).road = 'curve';
scenarios(5).disturbanceStart = 3; scenarios(5).disturbanceEnd = 3.6;
scenarios(5).lateralDisturbance = 0.65; scenarios(5).yawDisturbance = 0.035;
scenarios(6).name = 'Perception_low_confidence'; scenarios(6).road = 'curve';
scenarios(6).lowConfidenceStart = 3; scenarios(6).lowConfidenceEnd = 7;
end
