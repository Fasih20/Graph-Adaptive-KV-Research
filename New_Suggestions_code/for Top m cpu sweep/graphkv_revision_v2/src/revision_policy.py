"""Version 2 ablations: identical topology/scoring except the named intervention."""
from collections import defaultdict
import numpy as np
from adaptive_policies import Prediction, OfflineAdaptivePolicy, OnlineAdaptivePolicy, rank_paths, _blend

NAMES = ('no_prefetch', 'cosine', 'semantic_only', 'structure', 'two_hop',
         'offline', 'online')

def edges(similarity, documents, positions, m, structure=True):
    n = len(similarity)
    if m < 1:
        raise ValueError('M must be positive')
    top = [set(sorted((j for j in range(n) if j != i),
                      key=lambda j: (-float(similarity[i,j]), j))[:m]) for i in range(n)]
    result = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i+1,n):
            same = documents[i] == documents[j]
            distance = abs(positions[i]-positions[j]) if same else None
            semantic = j in top[i] or i in top[j]
            if semantic or (structure and same and distance == 1):
                f = (float(similarity[i,j]) if semantic else 0.,
                     1./(1+distance) if structure and same else 0.)
                result[i].append((j,*f)); result[j].append((i,*f))
    return result

class Policy(OnlineAdaptivePolicy):
    def __init__(self, name, raw, similarity, weights=(.7,.3), eta=.01):
        if name not in NAMES: raise ValueError(name)
        chosen = weights if name in ('offline','online') else ((1.,0.) if name == 'semantic_only' else (.7,.3))
        super().__init__(raw,chosen,eta)
        self.name, self.similarity = name, similarity

    def predict(self, primary, k):
        if self.name == 'no_prefetch': return Prediction([])
        if self.name == 'cosine':
            ids = sorted((j for j in range(len(self.similarity)) if j != primary),
                         key=lambda j: (-float(self.similarity[primary,j]),j))
            return Prediction(ids[:k], candidate_universe=set(ids))
        paths = defaultdict(list)
        for j,s,t in self.raw_edges[primary]:
            paths[j].append((s,t))
            if self.name not in ('semantic_only','structure'):
                for z,s2,t2 in self.raw_edges[j]:
                    if z != primary: paths[z].append(_blend((s,t),(s2,t2)))
        return rank_paths(paths,self.weights,k)

    def observe(self,target,prediction):
        if self.name == 'online': return super().observe(target,prediction)
        return OfflineAdaptivePolicy.observe(self,target,prediction)

def fit(train, dev, raw, sim, ks):
    grid=[]
    for a in np.linspace(0,1,21):
        p=Policy('offline',raw,sim,(a,1-a))
        hits=[e.target_chunk_id in p.predict(e.current_chunk_id,k).ids for e in train for k in ks]
        grid.append({'semantic':float(a),'recall':float(np.mean(hits))})
    best=max(grid,key=lambda r:(r['recall'],-abs(r['semantic']-.7),-r['semantic']))
    weights=(best['semantic'],1-best['semantic']); etas=[]
    for eta in (.01,.05,.1,.2,.5,1.):
        hits=[]
        for k in ks:
            p=Policy('online',raw,sim,weights,eta)
            for e in dev:
                prediction=p.predict(e.current_chunk_id,k)
                hits.append(e.target_chunk_id in prediction.ids)
                p.observe(e.target_chunk_id,prediction)
        etas.append({'eta':eta,'recall':float(np.mean(hits))})
    eta=max(etas,key=lambda r:(r['recall'],-abs(r['eta']-.1)))['eta']
    return {'weights':weights,'eta':eta,'train_grid':grid,'dev_eta':etas}
