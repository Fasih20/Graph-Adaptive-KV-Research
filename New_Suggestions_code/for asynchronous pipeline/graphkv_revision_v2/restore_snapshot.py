#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from revision_io import restore
p=argparse.ArgumentParser(description='Verify hashes and restore a downloaded snapshot into an empty directory.')
p.add_argument('snapshot'); p.add_argument('destination')
a=p.parse_args(); restore(a.snapshot,a.destination); print('Snapshot verified and restored.')
