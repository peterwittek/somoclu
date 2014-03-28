 
Somoclu MATLAB Extension Build Guide:

1. follow the instructions to build Somoclu itself at:
https://github.com/peterwittek/somoclu

2. Build MATLAB Extension by running: 
MEX_BIN="/usr/local/MATLAB/R2013a/bin/mex" ./makeMex.sh
where MEX_BIN is the path to the MATLAB installation mex binary

3. Then MexSomoclu.mexa64/MexSomoclu.mexa32 is generated for use, you can test by running the mex_interface_test.m

Version: 1.2
Maintainer: ShichaoGao<xgdgsc at gmail.com>
URL: http://peterwittek.github.io/somoclu/
BugReports: https://github.com/peterwittek/somoclu/issues
License: GPL-3
OS_type: unix
