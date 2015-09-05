function [sMap, sTrain] = somoclu_train(sMap, D, varargin)
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
%   'radiusCooling'       (string) Radius cooling strategy: linear or exponential (default: linear)
%   'scale0' 'alpha_ini'   (scalar) Starting learning rate (default: 0.1)
%   'scaleN'    (scalar)Finishing learning rate (default: 0.01)
%   'kernelType'   (string)  Kernel type
%   'mapType' 'shape' (string)  Map type: planar or toroid (default: planar)
%   'scaleCooling' 'alpha_type' (string)  Learning rate cooling strategy: linear or exponential (default: linear)
%   'gridType' 'lattice'  (string)  Grid type: square or hexagonal (default: square)
%   'compactSupport'  Compact support for map update (0: false, 1: true, default: 0)
%   'nEpoch' 'trainlen' (scalar)  Maximum number of epochs
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
     case 'sample_order', i=i+1; sample_order_type = varargin{i};
     case {'radius0', 'radius_ini'}, i=i+1; sTrain.radius_ini = varargin{i};
     case {'radiusN', 'radius_fin'}, i=i+1; sTrain.radius_fin = varargin{i};
     case {'scaleCooling', 'alpha_type'}, i=i+1; sTrain.alpha_type = varargin{i};
     case {'scale0', 'alpha_ini'}, i=i+1; sTrain.alpha_ini = varargin{i};
     case {'sTrain','train','som_train'}, i=i+1; sTrain = varargin{i};
      % unambiguous values
     case {'inv','linear','power'}, sTrain.alpha_type = varargin{i}; 
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
