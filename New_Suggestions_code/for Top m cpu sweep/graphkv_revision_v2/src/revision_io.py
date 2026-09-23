"""Atomic checkpoints and opt-in, verified, non-destructive Drive snapshots."""
import hashlib, json, os, shutil, subprocess, tempfile, time, zipfile
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()

def atomic(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,default=str); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def fingerprint(root,config):
    path=Path(root)/'revision_manifest.json'
    if path.exists() and json.loads(path.read_text())['fingerprint'] != digest(config):
        raise ValueError('Configuration/source changed. Use a NEW output directory; old results preserved.')
    atomic(path,{'fingerprint':digest(config),'config':config})

class Backup:
    def __init__(self,root,remote=None):
        self.root=Path(root); self.remote=remote
        if remote:
            if ':' not in remote or remote.startswith('-'): raise ValueError('Use rclone remote:path')
            if not shutil.which('rclone'): raise RuntimeError('Install/configure rclone before enabling backup')

    def save(self,label):
        if not self.remote: return
        stamp=f'{time.time_ns()}-{label}'
        with tempfile.TemporaryDirectory(prefix='graphkv-backup-') as td:
            archive=Path(td)/f'{stamp}.zip'; hashes={}
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for p in sorted(self.root.rglob('*')):
                    if not p.is_file() or p.is_symlink() or p.suffix == '.tmp': continue
                    # Only research-output formats; credentials/configs must live outside output.
                    if p.suffix not in ('.json','.jsonl','.csv','.npz','.npy','.txt','.log'): continue
                    rel=str(p.relative_to(self.root)); data=p.read_bytes()
                    hashes[rel]=hashlib.sha256(data).hexdigest(); z.writestr(rel,data)
                z.writestr('BACKUP_HASHES.json',json.dumps(hashes,indent=2))
            destination=self.remote.rstrip('/')+'/'+archive.name
            commands=[['rclone','copyto',str(archive),destination,'--retries','3'],
                      ['rclone','check',td,self.remote,'--one-way','--download']]
            for cmd in commands:
                result=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
                if result.returncode:
                    atomic(self.root/'backup_status.json',{'ok':False,'snapshot':stamp})
                    raise RuntimeError('Drive backup/verification failed; local checkpoint preserved. Fix rclone and resume.')
            atomic(self.root/'backup_status.json',{'ok':True,'snapshot':destination,'verified':True})
            print(f'Drive snapshot verified: {archive.name}',flush=True)

def restore(archive,destination):
    root=Path(destination)
    if root.exists() and any(root.iterdir()): raise ValueError('Restore into an empty directory')
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        hashes=json.loads(z.read('BACKUP_HASHES.json'))
        for name,expected in hashes.items():
            path=root/name
            if not path.resolve().is_relative_to(root.resolve()): raise ValueError('Unsafe ZIP path')
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('Corrupt snapshot: '+name)
        for name in hashes:
            path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(z.read(name))
