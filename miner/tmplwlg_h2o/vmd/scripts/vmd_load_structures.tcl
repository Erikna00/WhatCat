set dir "/home/erikna/compchem/WhatCat/miner/tmplwlg_h2o"

mol load pdb ${dir}/frame_1.pdb
animate read pdb ${dir}/frame_2.pdb
animate read pdb ${dir}/frame_3.pdb
animate read pdb ${dir}/frame_4.pdb
animate read pdb ${dir}/frame_5.pdb
animate read pdb ${dir}/frame_6.pdb
animate read pdb ${dir}/frame_7.pdb
animate read pdb ${dir}/frame_8.pdb
animate read pdb ${dir}/frame_9.pdb

after idle { 
  mol representation NewCartoon 
  mol delrep 0 top
  mol addrep top
  mol modcolor 0 top "ColorID" 8
} 

