#set dir "/home/erikna/whatcat/miner/work_dir/1be0"

mol load pdb ../data/1be0.pdb

after idle { 
  mol representation NewCartoon 
  mol delrep 0 top
  mol addrep top
  mol modcolor 0 top "ColorID" 8
} 

