#set dir "/home/erikna/compchem/WhatCat/miner/tmpcwqu_r32"

mol load pdb ../data/frame_14.pdb

after idle { 
  mol representation NewCartoon 
  mol delrep 0 top
  mol addrep top
  mol modcolor 0 top "ColorID" 8
} 

