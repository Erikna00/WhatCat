#set dir "/home/erikna/compchem/WhatCat/miner/tmplwlg_h2o"

mol load pdb ../data/frame_5.pdb

after idle { 
  mol representation NewCartoon 
  mol delrep 0 top
  mol addrep top
  mol modcolor 0 top "ColorID" 8
} 

