function spekgui(action,varargin)
% SPEKGUI
%
% spekgui  opens a window for spectral estimation.
%
% Import data by putting them into a "structure", write the name in the "Import"-box
% and push the button.
% Example:
%
% >> litedata.t=linspace(0,50,1001);
% >> litedata.x=sin(2*pi*litedata.t)+randn(1,1001)*0.5;
%
% The different spectral estimation methods are :
%
%   Periodogram: The usual periodogram. 
%   Bartlett: averaging over m periodogram without windows.
%   Welch: averaging using 'Hanning' windows of the m periodograms and 50% overlap.
%
% The covariance function is estimated from data or from
% the spectral density estimate.
%
% The estimates of the covariance function and spectral density
% is exported to Matlab with the "Export"-button.
%

if nargin<1
  action='init';
elseif ~strcmp(action,'init')
  if nargin>1
    f=varargin{1};
  else
    f=gcbf;
  end
end

switch action
  case 'init',
    initspek;
    f=gcf;
    
    data.covflength=100;
    data.specN=2^12;
    
    data.param=0;
    
    data.t=0:1;
    data.dt=data.t(2)-data.t(1);
    data.x=data.t*0;
    data.mx=mean(data.x);
    
    data.tau=0:data.covflength;
    data.r=data.tau*0;
    
    data.f=(0:data.specN-1)/data.specN/2;
    data.R=data.f*0;
    
    data.h_covflength=findobj(f,'Tag','covflength');
    set(data.h_covflength,'String',int2str(data.covflength));

    data.h_covf=findobj(f,'Tag','Covf-axis');
    data.h_covfdata=line(data.tau,data.r,'Parent',data.h_covf);
    line(0,0,'Parent',data.h_covf);

    data.h_spec=findobj(f,'Tag','Spec-axis');
    data.h_specdata=line(data.f,data.R,'Parent',data.h_spec);
    line(0,0,'Parent',data.h_spec);
  
    data.h_data=findobj(f,'Tag','Data-axis');
    data.h_datadata=line(data.t,data.x,'Parent',data.h_data);
    line(0,0,'Parent',data.h_data);
    
    data.zoom=1;
    data.zoomwidth=data.t(end)-data.t(1);
    data.h_zoom=findobj(f,'Tag','zoom');
    set(data.h_zoom,'Value',data.zoom);

    data.location=data.zoomwidth/2+data.t(1);
    data.h_location=findobj(f,'Tag','location');
    set(data.h_location,...
	'Min',data.zoomwidth/2+data.t(1),...
	'Max',data.zoomwidth/2+1e-9+data.t(1),...
	'Value',data.location);

    data.zoomspec=1;
    data.zoomwidthspec=1/2/data.dt;
    data.h_zoomspec=findobj(f,'Tag','zoomspec');
    set(data.h_zoomspec,'Value',data.zoomspec);

    data.locationspec=data.zoomwidthspec/2;
    data.h_locationspec=findobj(f,'Tag','locationspec');
    set(data.h_locationspec,...
	'Min',data.zoomwidthspec/2,...
	'Max',data.zoomwidthspec/2+1e-9,...
	'Value',data.locationspec);

    data.h_specmethod=findobj(f,'Tag','specmethod');
    data.h_covfmethod=findobj(f,'Tag','covfmethod');

    data.h_exportname=findobj(f,'Tag','exportname');
    set(data.h_exportname,'String','estdata');

    data.h_importname=findobj(f,'Tag','importname');
    set(data.h_importname,'String','data');

    data.h_dt=findobj(f,'Tag','dt');
    set(data.h_dt,'String',num2str(data.dt));

    data.h_param=findobj(f,'Tag','param');
    set(data.h_param,'String',num2str(data.param));

    data.scale=1;
    data.h_scale=findobj(f,'Tag','specscale');
    set(data.h_scale,'Value',data.scale);
    if data.scale==1
      set(data.h_spec,'YScale','linear')
    else
      set(data.h_spec,'YScale','log')
    end
    
    data.specmethod=0;

    set(f,'UserData',data);
    set(f,'handleVisibility','callback');
    
    setzoompos(f);
    setzoomposspec(f);
    spekgui('specmethod',f);
  case 'estimate',
    estimatespectrum(f);
    estimatecovf(f);
  case 'specmethod',
    data=get(f,'UserData');
    method=get(data.h_specmethod,'Value');
    if method~=data.specmethod
      switch method,
	case 1, % Periodogram
	  data.param=0;
	case 2, % Bartlett
	  if data.specmethod==3
	    data.param=data.param(1);
	  else
	    data.param=20;
	  end
	case 3, % Welch 
	  if data.specmethod==2
	    data.param=data.param(1);
	  else
	    data.param=20;
	  end
	case 4, % Tidsfönster
	  if data.specmethod==5
	    data.param=ceil(max(2,1./max(5e-4,data.param(1))));
	  else
	    data.param=max(ceil(length(data.x)/10),30);
	  end
	case 5, % Frekvensfönster
	  if data.specmethod==4
	    data.param=1./max(2,min(5e4,data.param(1)));
	  else
	    data.param=1./max(ceil(length(data.x)/10),30);
	  end
      end
      data.specmethod=method;
      set(data.h_param,'String',num2str(data.param));
      set(f,'UserData',data);
      estimatespectrum(f);
      estimatecovf(f);
    end
  case 'zoom',
    data=get(f,'UserData');
    data.zoom=get(data.h_zoom,'Value');
    if data.zoom==1
      data.zoomwidth=data.t(end)-data.t(1);
      data.location=data.zoomwidth/2+data.t(1);
    else
      data.zoomwidth=(data.t(end)-data.t(1))/2^(data.zoom-1)+data.t(1);
    end
    if data.location-data.zoomwidth/2<data.t(1)
      data.location=data.zoomwidth/2+data.t(1);
    elseif data.location+data.zoomwidth/2>data.t(end)
      data.location=data.t(end)-data.zoomwidth/2;
    end
    set(data.h_location,'Min',data.zoomwidth/2+data.t(1),...
	                'Max',data.t(end)-data.zoomwidth/2+1e-9,...
			'Value',data.location);
    set(f,'UserData',data);
    setzoompos(f);
  case 'zoomspec',
    data=get(f,'UserData');
    data.zoomspec=get(data.h_zoomspec,'Value');
    if data.zoomspec==1
      data.zoomwidthspec=1/2/data.dt;
      data.locationspec=data.zoomwidthspec/2;
    else
      data.zoomwidthspec=1/2/data.dt/2^(data.zoomspec-1);
    end
    if data.locationspec-data.zoomwidthspec/2<0
      data.locationspec=data.zoomwidthspec/2;
    elseif data.locationspec+data.zoomwidthspec/2>1/2/data.dt
      data.locationspec=1/2/data.dt-data.zoomwidthspec/2;
    end
    set(data.h_locationspec,'Min',data.zoomwidthspec/2,...
	                'Max',1/2/data.dt-data.zoomwidthspec/2+1e-9,...
			'Value',data.locationspec);
    set(f,'UserData',data);
    setzoomposspec(f);
  case 'location',
    data=get(f,'UserData');
    data.location=get(data.h_location,'Value');
    set(f,'UserData',data);
    setzoompos(f);
  case 'locationspec',
    data=get(f,'UserData');
    data.locationspec=get(data.h_locationspec,'Value');
    set(f,'UserData',data);
    setzoomposspec(f);
  case 'specscale',
    data=get(f,'UserData');
    data.scale=get(data.h_scale,'Value');
    set(f,'UserData',data);
    if data.scale==1
      set(data.h_spec,'YScale','linear')
    else
      set(data.h_spec,'YScale','log')
    end
  case 'export',
    data=get(f,'UserData');
    data.exportname=get(data.h_exportname,'String');
    set(f,'UserData',data);
    thedata.tau=data.tau;
    thedata.r=data.r;
    thedata.f=data.f;
    thedata.R=data.R;
    thedata.t=data.t;
    thedata.dt=data.dt;
    thedata.x=data.x;
    assignin('base',data.exportname,thedata);
  case 'import',
    data=get(f,'UserData');
    data.importname=get(data.h_importname,'String');
    evalin('base','global spekguitemp;');
    global spekguitemp;
    set(f,'UserData',data);
    er=0;
    evalin('base',['spekguitemp=' data.importname ';'],'er=1;');
    if er % Variable not found
       errordlg('Variable not found','Error')
       return;
    end
    data.x=spekguitemp.x(:)';
    data.mx=mean(data.x);
    if isfield(spekguitemp,'t')
      data.t=spekguitemp.t(:)';
    else
      data.t=0:length(spekguitemp.x)-1;
      if isfield(spekguitemp,'dt')
	data.t=data.t*spekguitemp.dt;
      end
    end
    data.dt=data.t(2)-data.t(1);
    set(data.h_dt,'String',num2str(data.dt))
    set(data.h_datadata,'XData',data.t,'YData',data.x);
    
    if isfield(spekguitemp,'tau')
      data.covflength=length(spekguitemp.tau)-1;
      set(data.h_covflength,'String',int2str(data.covflength))
    end
    data.tau=(0:data.covflength)*data.dt;
    data.r=data.tau*0;
    set(data.h_covfdata,'XData',data.tau,'YData',data.r);
    set(f,'UserData',data);
    spekgui dt
    makexdata(f);
    spekgui zoom
    spekgui location
    spekgui zoomspec
    spekgui estimate
  case 'dt'
    data=get(f,'UserData');
    value=str2num(get(data.h_dt,'String'));
    if isempty(value)
      set(data.h_dt,'String',num2str(data.dt));
    else
      olddt=data.dt;
      data.dt=max(1e-9,value(1));
      set(data.h_dt,'String',num2str(data.dt));
      set(f,'UserData',data);
      makexdata(f);
      adjustdt(f,olddt);
      spekgui zoom
      spekgui location
      spekgui zoomspec
      spekgui locationspec
    end
  case 'param'
    data=get(f,'UserData');
    value=str2num(get(data.h_param,'String'));
    if isempty(value)
      set(data.h_param,'String',num2str(data.param));
    else
      data.param=value;
      set(data.h_param,'String',num2str(data.param));
      set(f,'UserData',data);
      spekgui estimate
    end
  case 'covflength'
    data=get(f,'UserData');
    value=str2num(get(data.h_covflength,'String'));
    if isempty(value)
      set(data.h_covflength,'String',int2str(data.covflength));
    else
      data.covflength=max(0,ceil(value(1)));
      set(data.h_covflength,'String',int2str(data.covflength));
      set(f,'UserData',data);
      makexdata(f);
      estimatecovf(f);
    end
  case 'close'
    close(f)
  otherwise,
    disp(['Unknown action: ' action])
end


function adjustdt(f,olddt)
data=get(f,'UserData');
data.R=data.R/olddt*data.dt;
set(data.h_specdata,'YData',data.R);
set(f,'UserData',data);

function makexdata(f)
data=get(f,'UserData');
data.t=(0:length(data.x)-1)*data.dt;
set(data.h_datadata,'XData',data.t);
data.tau=(0:data.covflength)*data.dt;
set(data.h_covfdata,'XData',data.tau);
set(data.h_covf,'XLim',[0 data.tau(end)]);
data.f=(0:data.specN-1)/data.specN/2/data.dt;
set(data.h_specdata,'XData',data.f);
set(data.h_spec,'XLim',[0 1/2/data.dt]);
set(f,'UserData',data);


function setzoompos(f);
data=get(f,'UserData');
set(data.h_data,'XLim',data.location+[-1 1]*data.zoomwidth/2);

function setzoomposspec(f);
data=get(f,'UserData');
set(data.h_spec,'XLim',data.locationspec+[-1 1]*data.zoomwidthspec/2);


function estimatespectrum(f)
data=get(f,'UserData');
data.R=0*data.f;
switch data.specmethod
  case 1, % Periodogram
    data.R=fourier(data.x-data.mx,data.specN);
    data.R=data.R.*conj(data.R)/length(data.x)*data.dt;
    data.R(1)=0;
  case 2, % Bartlett
    if length(data.param)>1
      p=data.param(2);
    else
      p=0;
    end
    data.R=welch(data.x,data.dt,ceil(data.param(1)),p,...
	data.f,data.specN,'rect');
  case 3, % Welch 
    if length(data.param)>1
      p=data.param(2);
    else
      p=0.5;
    end
    data.R=welch(data.x,data.dt,ceil(data.param(1)),p,...
	data.f,data.specN,'hanning');
  case 4, % Tidsfönster
    M=ceil(max(2,min(5e4,data.param(1))));
    data.R=spa(data.x(:)-mean(data.x),M,2*pi*data.f(:),[],data.dt);
    % ident ver 5.x is incompatible with 4.x. BAD Mathworks!
    ident_ver = ver('ident');
    if (str2num(ident_ver.Version(1))>4)
      data.R = squeeze(get(data.R,'SpectrumData'))';
    else
      data.R=data.R(2:end,2)';
    end
  case 5,
    M=ceil(max(2,1./max(5e-4,data.param(1))));
    data.R=spa(data.x(:)-mean(data.x),M,2*pi*data.f(:),[],data.dt);
    % ident ver 5.x is incompatible with 4.x. BAD Mathworks!
    ident_ver = ver('ident');
    if (str2num(ident_ver.Version(1))>4)
      data.R = squeeze(get(data.R,'SpectrumData'))';
    else
      data.R=data.R(2:end,2)';
    end
  otherwise,
    disp(['Unknown spectral estimation method, ' int2str(data.specmethod) '.'])
end
set(data.h_specdata,'YData',data.R);
set(f,'UserData',data);


function F=fourier(f,N)
P=nextpow2(2*N);
p=nextpow2(length(f));
n=2^max(P,p);
p=max(P,p)-P;
F=fft(f,n);
F=F((0:N-1)*(2^p)+1);


function estimatecovf(f)
data=get(f,'UserData');
method=get(data.h_covfmethod,'Value');
data.r=0*data.tau;
switch method
  case 1, % Från data
    if length(data.x)-1<data.covflength
      data.r=conv(data.x-data.mx,fliplr(data.x)-data.mx);
      data.r=data.r(end-length(data.x)+1:end);
      data.r=[data.r zeros(1,data.covflength-length(data.r)+1)];
    else
      n=data.covflength*2;
      for k=1:ceil(length(data.x)/n)
	i=n*(k-1)+1;
	j=min(length(data.x),i+n-1);
	y=data.x(i:j)-data.mx;
        r=conv(y,fliplr(y));
        r=r(end-length(y)+1:end);
	if length(r)>=data.covflength+1
	  data.r=data.r+r(1:data.covflength+1);
	else
	  data.r(1:length(r))=data.r(1:length(r))+r;
	end
      end
    end
    data.r=data.r/length(data.x);
  case 2, % Från spektrum
    for k=1:length(data.tau)
      data.r(k)=data.R(1)+...
	  2*sum(cos(2*pi*data.f(2:end)*data.tau(k)).*data.R(2:end));
    end
    data.r=data.r*(data.f(2)-data.f(1));
  otherwise,
    disp(['Unknown covariance estimation method, ' int2str(method) '.'])
end
set(data.h_covfdata,'YData',data.r);
set(f,'UserData',data);


