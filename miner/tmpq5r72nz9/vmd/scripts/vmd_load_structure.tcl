#set dir "/home/erikna/compchem/WhatCat/miner/tmpq5r72nz9"

mol load pdb ../data/frame_15.pdb

after idle { 
  mol representation NewCartoon 
  mol delrep 0 top
  mol addrep top
  mol modcolor 0 top "ColorID" 8
} 

