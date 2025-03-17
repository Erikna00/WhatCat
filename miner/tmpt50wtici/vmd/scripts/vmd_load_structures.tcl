set dir "/home/erikna/compchem/WhatCat/miner/tmpt50wtici"

mol load pdb ${dir}/frame_1.pdb
animate read pdb ${dir}/frame_2.pdb
animate read pdb ${dir}/frame_3.pdb
animate read pdb ${dir}/frame_4.pdb
animate read pdb ${dir}/frame_5.pdb
animate read pdb ${dir}/frame_6.pdb
animate read pdb ${dir}/frame_7.pdb
animate read pdb ${dir}/frame_8.pdb
animate read pdb ${dir}/frame_9.pdb
animate read pdb ${dir}/frame_10.pdb
animate read pdb ${dir}/frame_11.pdb
animate read pdb ${dir}/frame_12.pdb
animate read pdb ${dir}/frame_13.pdb
animate read pdb ${dir}/frame_14.pdb
animate read pdb ${dir}/frame_15.pdb
animate read pdb ${dir}/frame_16.pdb
animate read pdb ${dir}/frame_17.pdb
animate read pdb ${dir}/frame_18.pdb
animate read pdb ${dir}/frame_19.pdb
animate read pdb ${dir}/frame_20.pdb
animate read pdb ${dir}/frame_21.pdb
animate read pdb ${dir}/frame_22.pdb
animate read pdb ${dir}/frame_23.pdb
animate read pdb ${dir}/frame_24.pdb
animate read pdb ${dir}/frame_25.pdb
animate read pdb ${dir}/frame_26.pdb
animate read pdb ${dir}/frame_27.pdb
animate read pdb ${dir}/frame_28.pdb

after idle { 
  mol representation NewCartoon 
  mol delrep 0 top
  mol addrep top
  mol modcolor 0 top "ColorID" 8
} 

