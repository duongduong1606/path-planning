function summary = run_tcn_closed_loop_scenarios(weightsFile, outputDirectory)
%RUN_TCN_CLOSED_LOOP_SCENARIOS Compare raw TCN with a Frenet lane-center baseline.
% The safety shield is deliberately excluded from this model-only evaluation.
if nargin < 1
    root = fileparts(fileparts(mfilename('fullpath')));
    weightsFile = fullfile(root,'artifacts','tcn_lane_safe','lateral_tcn_weights.mat');
end
if nargin < 2
    root = fileparts(fileparts(mfilename('fullpath')));
    outputDirectory = fullfile(root,'artifacts','closed_loop');
end
if ~exist(outputDirectory,'dir'), mkdir(outputDirectory); end
data = load(weightsFile); scenarios = tcn_scenario_contract();
planners = {'TCN_raw','Frenet_baseline'};
rows = struct([]); traces = struct(); row = 0;
for s = 1:numel(scenarios)
    for p = 1:numel(planners)
        fprintf('[%d/%d] %s | %s\n',(s-1)*2+p,numel(scenarios)*2, ...
            scenarios(s).name,planners{p});
        trace = simulateOne(scenarios(s),planners{p},data);
        metric = summarize(trace,scenarios(s)); row = row+1;
        rows(row).Scenario = scenarios(s).name;
        rows(row).Planner = planners{p};
        names = fieldnames(metric);
        for k=1:numel(names), rows(row).(names{k})=metric.(names{k}); end
        traces.(matlab.lang.makeValidName([scenarios(s).name '_' planners{p}]))=trace;
    end
end
summary = struct2table(rows);
writetable(summary,fullfile(outputDirectory,'summary.csv'));
save(fullfile(outputDirectory,'traces.mat'),'traces','scenarios','-v7.3');
writeJson(summary,fullfile(outputDirectory,'summary.json'));
plotComparison(traces,scenarios,outputDirectory);
disp(summary);
end

function trace = simulateOne(scene,planner,data)
dt=.05; time=(0:dt:scene.duration).'; n=numel(time); L=2.8;
[yc0,psiRoad0]=roadGeometry(0,scene.road);
state=[0;yc0+scene.initialOffset;psiRoad0+deg2rad(scene.initialHeadingDeg);scene.speed];
feature=makeFeatures(state,0,0,0,1,scene.road);
history=repmat(feature,40,1); delta=0; previousDelta=0; previousPath=zeros(1,51);
trace.time=time; trace.state=zeros(n,4); trace.control=zeros(n,2);
trace.offset=zeros(n,1); trace.path=zeros(n,51); trace.terminal=zeros(n,4);
trace.confidence=zeros(n,1); trace.inferenceTime=zeros(n,1);
trace.boundaryViolation=false(n,1); trace.pathJump=zeros(n,1);
for i=1:n
    t=time(i); low=t>=scene.lowConfidenceStart && t<=scene.lowConfidenceEnd;
    confidence=1; validity=1; if low, confidence=.1; validity=.5; end
    yawRate=state(4)/L*tan(delta); lateralAccel=state(4)^2/L*tan(delta);
    steeringRate=(delta-previousDelta)/dt;
    feature=makeFeatures(state,yawRate,lateralAccel,delta,confidence,scene.road);
    feature(15)=steeringRate; feature(16)=validity;
    history=[history(2:end,:);feature];
    if strcmp(planner,'TCN_raw')
        normalized=normalizeFromStruct(history,data.normalization);
        timer=tic; terminal=lateral_tcn_forward(normalized,data.weights);
        trace.inferenceTime(i)=toc(timer);
    else
        terminal=frenetTerminal(feature,state(4),2.5);
    end
    [path,velocity]=relativeQuintic(terminal,2.5,51);
    look=round(.55/2.5*50)+1; yRef=path(look);
    headingRef=atan2(velocity(look),max(state(4),1));
    kct=1.25; kh=1.65; ayMax=3.8;
    steeringLimit=min(.5,atan(ayMax*L/max(state(4)^2,25)));
    rawDelta=kh*headingRef+atan2(kct*yRef,state(4)+1.5);
    rawDelta=max(-steeringLimit,min(steeringLimit,rawDelta));
    previousDelta=delta;
    delta=max(delta-.8*dt,min(delta+.8*dt,rawDelta));
    acceleration=.85*(scene.speed-state(4)); acceleration=max(-5,min(3,acceleration));
    latDist=0; yawDist=0;
    if t>=scene.disturbanceStart && t<=scene.disturbanceEnd
        latDist=scene.lateralDisturbance; yawDist=scene.yawDisturbance;
    end
    beta=atan(.5*tan(delta)); v=max(0,state(4));
    derivative=[v*cos(state(3)+beta);v*sin(state(3)+beta)+latDist; ...
        v*cos(beta)*tan(delta)/L+yawDist;acceleration];
    trace.state(i,:)=state.'; trace.control(i,:)=[acceleration delta];
    [roadY,~]=roadGeometry(state(1),scene.road);
    trace.offset(i)=state(2)-roadY; trace.path(i,:)=path;
    trace.terminal(i,:)=terminal; trace.confidence(i)=confidence;
    trace.boundaryViolation(i)=pathViolation(path,feature,state(4));
    if i>1, trace.pathJump(i)=max(abs(path-previousPath)); end
    previousPath=path; state=state+dt*derivative;
end
end

function feature = makeFeatures(ego,yawRate,lateralAccel,steering,confidence,road)
xSample=linspace(5,60,40); c=cos(ego(3)); s=sin(ego(3)); width=3.7;
coeff=zeros(2,4);
for side=1:2
    signSide=3-2*side; % left +1, right -1
    worldX=ego(1)+xSample;
    [center,~]=roadGeometry(worldX,road);
    worldY=center+signSide*width/2;
    dx=worldX-ego(1); dy=worldY-ego(2);
    localX=c*dx+s*dy; localY=-s*dx+c*dy;
    fit=polyfit(localX,localY,3); coeff(side,:)=fliplr(fit);
end
feature=[coeff(1,:) coeff(2,:) confidence confidence ego(4) yawRate ...
    lateralAccel steering 0 1];
end

function terminal = frenetTerminal(feature,speed,horizon)
x=max(speed,0)*horizon; center=.5*(feature(1:4)+feature(5:8));
y=polyval(fliplr(center),x);
dyDx=center(2)+2*center(3)*x+3*center(4)*x^2;
d2yDx2=2*center(3)+6*center(4)*x;
terminal=[y dyDx*speed d2yDx2*speed^2 0];
end

function normalized = normalizeFromStruct(history,stats)
low=reshape(double(stats.clip_low),1,[]); high=reshape(double(stats.clip_high),1,[]);
mu=reshape(double(stats.mean),1,[]); sigma=reshape(double(stats.std),1,[]);
normalized=(min(max(double(history),low),high)-mu)./sigma;
end

function [path,velocity] = relativeQuintic(terminal,horizon,samples)
A=[horizon^3 horizon^4 horizon^5;3*horizon^2 4*horizon^3 5*horizon^4; ...
    6*horizon 12*horizon^2 20*horizon^3];
a=A\double(terminal(1:3)).'; t=linspace(0,horizon,samples);
path=a(1)*t.^3+a(2)*t.^4+a(3)*t.^5;
velocity=3*a(1)*t.^2+4*a(2)*t.^3+5*a(3)*t.^4;
end

function violation = pathViolation(path,feature,speed)
x=linspace(0,max(speed,0)*2.5,numel(path));
left=polyval(fliplr(feature(1:4)),x); right=polyval(fliplr(feature(5:8)),x);
violation=any(path>left | path<right);
end

function [y,psi] = roadGeometry(x,road)
if strcmp(road,'straight')
    y=zeros(size(x)); slope=zeros(size(x));
else
    y=3*sin(x/170)+1.2*sin(x/65);
    slope=3/170*cos(x/170)+1.2/65*cos(x/65);
end
psi=atan(slope);
end

function metric = summarize(trace,scene)
dt=median(diff(trace.time)); offset=trace.offset; delta=trace.control(:,2); v=trace.state(:,4);
accel=v.^2.*tan(delta)/2.8; jerk=[0;diff(accel)/dt]; steeringRate=[0;diff(delta)/dt];
departure=abs(offset)>1.85; recoveryIndex=find(abs(offset)>.2,1,'last');
metric.LaneDepartureFraction=mean(departure);
metric.RMSLateralOffset=sqrt(mean(offset.^2)); metric.MaxLateralOffset=max(abs(offset));
metric.RecoveryTime=0; if ~isempty(recoveryIndex), metric.RecoveryTime=trace.time(recoveryIndex); end
metric.MaxLateralAcceleration=max(abs(accel)); metric.MaxLateralJerk=max(abs(jerk));
metric.MaxSteeringRate=max(abs(steeringRate)); metric.PathJumpP95=prctile(trace.pathJump,95);
metric.RawBoundaryViolationFraction=mean(trace.boundaryViolation);
valid=trace.inferenceTime(trace.inferenceTime>0);
metric.InferenceTimeMean=0; metric.InferenceTimeP95=0;
if ~isempty(valid)
    metric.InferenceTimeMean=mean(valid); metric.InferenceTimeP95=prctile(valid,95);
end
metric.LowConfidenceDuration=max(0,scene.lowConfidenceEnd-scene.lowConfidenceStart);
end

function writeJson(tableValue,path)
records=table2struct(tableValue); text=jsonencode(records);
fid=fopen(path,'w'); cleanup=onCleanup(@()fclose(fid)); fprintf(fid,'%s',text);
end

function plotComparison(traces,scenarios,outputDirectory)
fig=figure('Visible','off','Color','w','Position',[50 50 1250 850]);
for s=1:numel(scenarios)
    subplot(3,2,s); hold on;
    a=traces.(matlab.lang.makeValidName([scenarios(s).name '_TCN_raw']));
    b=traces.(matlab.lang.makeValidName([scenarios(s).name '_Frenet_baseline']));
    plot(a.time,a.offset,'LineWidth',1.25); plot(b.time,b.offset,'--','LineWidth',1.15);
    yline(1.85,'r:'); yline(-1.85,'r:'); grid on; title(strrep(scenarios(s).name,'_',' '));
    xlabel('Time (s)'); ylabel('Lane offset (m)'); legend('TCN raw','Frenet','Location','best');
end
saveas(fig,fullfile(outputDirectory,'scenario_comparison.png')); close(fig);
end
