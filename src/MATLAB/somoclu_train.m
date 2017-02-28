function [sMap, sTrain, globalBmus, uMatrix] = somoclu_train(sMap, D, varargin)
%somoclu_train  Use somoclu to train the Self-Organizing Map.
%
% [sM,sT] = somoclu_train(sM, D, [[argID,] value, ...])
%
%  Input and output arguments ([]'s are optional): 
%   sM      (struct) map struct, the trained and updated map is returned
%           (matrix) codebook matrix of a self-organizing map
%                    size munits x dim or  msize(1) x ... x msize(k) x dim
%                    The trained map codebook is returned.
%   D       (struct) training data; data struct
%           (matrix) training data, size dlen x dim
%   [argID, (string) See below. The values which are unambiguous can 
%    value] (varies) be given without the preceeding argID.
%
%   sT      (struct) learning parameters used during the training
%
% Here are the valid argument IDs and corresponding values. The values which
% are unambiguous (marked with '*') can be given without the preceeding argID.
%   'msize'       (vector) map size
%   'radius0'  'radius_ini'    (scalar) Start radius (default: half of the map in direction min(x,y))
%   'radiusN' 'radius_fin' (scalar) End radius (default: 1)
%   'radius'      (vector) neighborhood radiuses, length 1, 2 
%   'radiusCooling'       (string) Radius cooling strategy: linear or exponential (default: linear)
%   'scale0' 'alpha_ini'   (scalar) Starting learning rate (default: 0.1)
%   'scaleN'    (scalar)Finishing learning rate (default: 0.01)
%   'kernelType'   (string)  Kernel type
%   'mapType' 'shape' (string)  Map type: planar or toroid (default: planar)
%   'scaleCooling' 'alpha_type' (string)  Learning rate cooling strategy: linear or exponential (default: linear)
%   'gridType' 'lattice'  (string)  Grid type: square or hexagonal (default: square)
%   'compactSupport'  Compact support for map update (0: false, 1: true, default: 0)
%   'neighborhood'  Neighborhood function (bubble or gaussian, default: gaussian)
%   'sdtCoeff' Coefficient in the Gaussian neighborhood function exp(-||x-y||^2/(2*(coeff*radius)^2)) (default: 0.5)
%   'nEpoch' 'trainlen' (scalar)  Maximum number of epochs
%   'sTrain','som_train '  = 'train'
%   'verbose'   Verbosity level: 0, 1, or 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Check arguments

error(nargchk(2, Inf, nargin));  % check the number of input arguments

% map 
struct_mode = isstruct(sMap);
if struct_mode, 
  sTopol = sMap.topol;
else  
  orig_size = size(sMap);
  if ndims(sMap) > 2, 
    si = size(sMap); dim = si(end); msize = si(1:end-1);
    M = reshape(sMap,[prod(msize) dim]);
  else
    msize = [orig_size(1) 1]; 
    dim = orig_size(2);
  end
  sMap   = som_map_struct(dim,'msize',msize);
  sTopol = sMap.topol;
end
[munits dim] = size(sMap.codebook);

% data
if isstruct(D), 
  data_name = D.name; 
  D = D.data; 
else 
  data_name = inputname(2); 
end
D = D(find(sum(isnan(D),2) < dim),:); % remove empty vectors from the data
[dlen ddim] = size(D);                % check input dimension
if dim ~= ddim, error('Map and data input space dimensions disagree.'); end

% varargin
sTrain = som_set('som_train','algorithm','seq','neigh', ...
		 sMap.neigh,'mask',sMap.mask,'data_name',data_name);
radius     = [];
alpha      = [];
tracking   = 1;
tlen_type  = 'epochs';

sTopol.lattice = 'square';
sTopol.shape = 'planar';
sTrain.radius_ini = 0;
sTrain.radius_fin = 1;
sTrain.radius_cooling = 'linear';
sTrain.alpha_type = 'linear';
sTrain.kernel_type = 0;
sTrain.compact_support = true;
sTrain.neighborhood = 'gaussian';
sTrain.scale0 = 0.1
sTrain.scaleN = 0.01
sTrain.stdCoeff = 0.5
sTrain.verbose = 0

i=1; 
while i<=length(varargin), 
  argok = 1; 
  if ischar(varargin{i}), 
    switch varargin{i}, 
     % argument IDs
     case 'msize', i=i+1; sTopol.msize = varargin{i}; 
     case {'gridType', 'lattice'}, i=i+1; sTopol.lattice = varargin{i};
     case {'mapType', 'shape'}, i=i+1; sTopol.shape = varargin{i};
     case {'nEpoch', 'trainlen'}, i=i+1; sTrain.trainlen = varargin{i};
     case {'radius0', 'radius_ini'}, i=i+1; sTrain.radius_ini = varargin{i};
     case 'radiusCooling', i=i+1; sTrain.radius_cooling = varargin{i};
     case {'radiusN', 'radius_fin'}, i=i+1; sTrain.radius_fin = varargin{i};
     case 'radius', 
      i=i+1; 
      l = length(varargin{i}); 
      if l==1, 
        sTrain.radius_ini = varargin{i}; 
      else 
        sTrain.radius_ini = varargin{i}(1); 
        sTrain.radius_fin = varargin{i}(end);
%         if l>2, radius = varargin{i}; tlen_type = 'samples'; end
      end 
     case {'scaleCooling', 'alpha_type'}, i=i+1; sTrain.alpha_type = varargin{i};
     case 'neighborhood', i=i+1; sTrain.neighborhood = varargin{i};
     case 'stdCoeff', i=i+1; sTrain.stdCoeff = varargin{i};
     case {'scale0', 'alpha_ini'}, i=i+1; sTrain.scale0 = varargin{i};
     case 'scaleN', i=i+1; sTrain.scaleN = varargin{i};
     case {'sTrain','train','som_train'}, i=i+1; sTrain = varargin{i};
     case 'kernelType', i=i+1; sTrain.kernel_type = varargin{i};
     case 'verbose', i=i+1; sTrain.verbose = varargin{i};
     case 'compactSupport', i=i+1; sTrain.compact_support = varargin{i};
      % unambiguous values
%      case {'inv','linear','power'}, sTrain.alpha_type = varargin{i}; 
     case {'hexa','rect'}, sTopol.lattice = varargin{i};
     case {'sheet','cyl','toroid'}, sTopol.shape = varargin{i}; 
     otherwise argok=0; 
    end
  elseif isstruct(varargin{i}) && isfield(varargin{i},'type'), 
    switch varargin{i}(1).type, 
     case 'som_topol', 
      sTopol = varargin{i}; 
      if prod(sTopol.msize) ~= munits, 
        error('Given map grid size does not match the codebook size.');
      end
     case 'som_train', sTrain = varargin{i};
     otherwise argok=0; 
    end
  else
    argok = 0; 
  end
  if ~argok, 
    disp(['(som_seqtrain) Ignoring invalid argument #' num2str(i+2)]); 
  end
  i = i+1; 
end
if strcmp(sTrain.neighborhood, 'gaussian')
    sTrain.gaussian = 1;
else
    sTrain.gaussian = 0;
end
[sMap.codebook, globalBmus, uMatrix] = MexSomoclu(D, sTrain.trainlen, ...
sTopol.msize(1), sTopol.msize(2), ...
sTrain.radius_ini, sTrain.radius_fin, ...
sTrain.radius_cooling,  sTrain.scale0, sTrain.scaleN, ...
sTrain.alpha_type, ...
sTrain.kernel_type, sTopol.shape, sTopol.lattice, ...
sTrain.compact_support, sTrain.gaussian, sTrain.stdCoeff, sMap.codebook, ...
sTrain.verbose);
rowNums=colon(1,size(globalBmus,1))'
globalBmus = [rowNums globalBmus]
sTrain = som_set(sTrain,'time',datestr(now,0));
