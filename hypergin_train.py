# ═══════════════════════════════════════════════════════════════════════════
#  HyperGIN — Final 3-Module Unified Pipeline
#
#  Input Image
#       │
#  ─────────────────────────────────────────────
#  Module 1: Forensic Feature Extraction
#  ─────────────────────────────────────────────
#  ResNet-50 ──────────────┐
#                          │
#  SRM Filters + CNN ──────┤
#                          ▼
#       SRM-guided Multi-head Cross-Attention (h=4)
#       Q, K from F_srm  |  V from F_app
#               │
#       Residual Fusion (+ F_app)
#               │
#       L2 Normalisation → F_norm
#       Skip connections  → S0, S1, S2
#  ─────────────────────────────────────────────
#  Module 2: HyperGIN Block × 2
#  ─────────────────────────────────────────────
#  Adaptive 2-Criteria Hypergraph
#    h_feat = W_feat · F_norm  (feature similarity)
#    h_srm  = W_srm  · F_srm  (SRM consistency)
#    a      = softmax(v·tanh(W·F_norm)) (suspicion)
#    H = softmax(w1·h_feat + w2·h_srm)
#               │
#  Attention-weighted HGNN × 2
#    F_hyp = H^T·(a⊙X)/D_e
#    X^(l) = LN(ReLU(Θ·H·F_hyp) + X^(l-1))
#               │
#  Displacement-aware GIN
#    d* = dominant copy-move displacement
#    W_nm = exp(-||d_nm - d*||² / 2σ²), σ learnable
#    F_gin = LN(MLP((1+ε)X + A_w·X))
#               │
#  Adaptive Forensic Gate
#    g = sigmoid(W_g·[X||GIN_msg||H_msg])
#    F_out = LN(MLP(g⊙(X+GIN+H)) + X)
#  ─────────────────────────────────────────────
#  Module 3: U-Net Decoder
#  ─────────────────────────────────────────────
#  Decoder Block 3 (16→32) + Skip S2
#  Decoder Block 2 (32→64) + Skip S1
#  Decoder Block 1 (64→128) + Skip S0
#  Bilinear Upsample (128→256)
#  1×1 Conv → Sigmoid → M_pred ∈ [0,1]^(H×W)
#
#  Dataset : /kaggle/input/datasets/tusharchauhan1898/comofod/CoMoFoD_small_v2
#  Output  : /kaggle/working/hypergin_outputs/
# ═══════════════════════════════════════════════════════════════════════════

import os, re, time, csv
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────
ROOT       = '/kaggle/input/datasets/tusharchauhan1898/comofod/CoMoFoD_small_v2'
OUT_DIR    = '/kaggle/working/hypergin_outputs'
MODEL_PT   = os.path.join(OUT_DIR, 'hypergin_best.pt')
TRAIN_CKPT = os.path.join(OUT_DIR, 'train_checkpoint.pt')
QUAL_DIR   = os.path.join(OUT_DIR, 'qualitative')
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(QUAL_DIR, exist_ok=True)

CFG = {
    'img_size'      : 256,
    'embed_dim'     : 256,      # C
    'num_heads'     : 4,        # multi-head attention
    'num_hyperedges': 16,       # K
    'top_k'         : 10,       # GIN similarity graph
    'num_layers'    : 2,        # HyperGIN block repetitions
    'batch_size'    : 4,
    'lr'            : 1e-4,
    'weight_decay'  : 1e-4,
    'epochs'        : 100,
    'patience'      : 25,
    'pos_weight'    : 5.0,
    'dice_w'        : 0.5,
    'thr_min'       : 0.20,
    'thr_max'       : 0.75,
    'thr_steps'     : 25,
    'val_ratio'     : 0.10,
    'test_ratio'    : 0.10,
    'seed'          : 42,
    'num_workers'   : 2,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Init] Device : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[Init] GPU    : {torch.cuda.get_device_name(0)}")
torch.manual_seed(CFG['seed']); np.random.seed(CFG['seed'])


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════

def parse_all_comofod(root):
    samples=[]; img_re=re.compile(r'^(\d+)_F.*\.(png|jpg|jpeg|bmp)$',re.IGNORECASE)
    def _scan(folder):
        files=sorted(os.listdir(folder)); fset=set(files)
        for fname in files:
            m=img_re.match(fname)
            if not m: continue
            num=m.group(1); mask=f"{num}_B.png"
            if mask in fset:
                samples.append({'img_path':os.path.join(folder,fname),
                                'mask_path':os.path.join(folder,mask),
                                'base_id':num})
    top=os.listdir(root)
    if any(os.path.isdir(os.path.join(root,e)) for e in top):
        for e in sorted(top):
            sub=os.path.join(root,e)
            if os.path.isdir(sub): _scan(sub)
    else: _scan(root)
    print(f"[Dataset] Total={len(samples)} | Base IDs={len(set(s['base_id'] for s in samples))}")
    return samples

def split_samples(samples, cfg):
    base_ids=sorted(set(s['base_id'] for s in samples))
    rng=np.random.default_rng(cfg['seed'])
    idx=rng.permutation(len(base_ids)).tolist()
    nt=max(1,int(len(idx)*cfg['test_ratio']))
    nv=max(1,int(len(idx)*cfg['val_ratio']))
    test_ids =set(base_ids[i] for i in idx[:nt])
    val_ids  =set(base_ids[i] for i in idx[nt:nt+nv])
    train_ids=set(base_ids[i] for i in idx[nt+nv:])
    train_s,val_s,test_s=[],[],[]
    for s in samples:
        bid=s['base_id']
        if   bid in train_ids: train_s.append(s)
        elif bid in val_ids:   val_s.append(s)
        elif bid in test_ids:  test_s.append(s)
    print(f"[Dataset] train={len(train_s)} val={len(val_s)} test={len(test_s)}")
    return train_s, val_s, test_s

class FullImageDataset(Dataset):
    def __init__(self, samples, cfg, augment=False):
        self.samples=samples; self.cfg=cfg; self.augment=augment
        self.to_tensor=transforms.Compose([
            transforms.Resize((cfg['img_size'],cfg['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
        self.color_aug=transforms.ColorJitter(
            brightness=0.3,contrast=0.3,saturation=0.2,hue=0.05) if augment else None
        print(f"  images={len(samples)} (lazy)")
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        s=self.samples[idx]
        img=cv2.imread(s['img_path']); msk=cv2.imread(s['mask_path'],cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            return self.__getitem__((idx+1)%len(self.samples))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if self.augment:
            if np.random.rand()>0.5: img=img[:,::-1,:].copy(); msk=msk[:,::-1].copy()
            if np.random.rand()>0.5: img=img[::-1,:,:].copy(); msk=msk[::-1,:].copy()
        img_pil=Image.fromarray(img)
        if self.color_aug: img_pil=self.color_aug(img_pil)
        img_t=self.to_tensor(img_pil)
        sz=self.cfg['img_size']
        msk_r=cv2.resize(msk,(sz,sz),interpolation=cv2.INTER_NEAREST)
        return img_t, torch.tensor((msk_r>127).astype(np.float32)).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 1 — Forensic Feature Extraction + Cross-Attention
# ═══════════════════════════════════════════════════════════════════════════

class SRMLayer(nn.Module):
    """3 fixed SRM high-pass filters → forensic noise residual."""
    SRM_KERNELS = torch.tensor([
        [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],
        [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
        [[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],
         [-1,-1,-1,-1,-1]]], dtype=torch.float32)/12.0
    def __init__(self,out_ch=128):
        super().__init__()
        self.register_buffer('w',self.SRM_KERNELS.unsqueeze(1))
        self.proj=nn.Sequential(
            nn.Conv2d(3,out_ch,3,padding=1,stride=2),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,stride=2),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,stride=2),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
    def forward(self,x):
        gray=0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]; gray=gray*0.229+0.456
        return self.proj(F.conv2d(gray,self.w,padding=2))


class MultiHeadForensicAttention(nn.Module):
    """
    SRM-guided Multi-head Cross-Attention.
    Q, K from F_srm — forensic noise guides attention
    V    from F_app — appearance features enhanced
    h=4 heads: each captures different forensic signal type
    Output: L2_norm(MultiHead(Q,K,V) + F_app)
    """
    def __init__(self,C,num_heads=4):
        super().__init__()
        self.h    = num_heads
        self.d    = C // num_heads
        self.W_q  = nn.Linear(C,C,bias=False)
        self.W_k  = nn.Linear(C,C,bias=False)
        self.W_v  = nn.Linear(C,C,bias=False)
        self.W_o  = nn.Linear(C,C,bias=False)
        self.scale= self.d ** -0.5

    def forward(self,x_app,x_srm):
        N=x_app.size(0); h=self.h; d=self.d
        Q=self.W_q(x_srm).view(N,h,d).transpose(0,1)  # (h,N,d)
        K=self.W_k(x_srm).view(N,h,d).transpose(0,1)  # (h,N,d)
        V=self.W_v(x_app).view(N,h,d).transpose(0,1)  # (h,N,d)
        A  =torch.softmax(torch.bmm(Q,K.transpose(1,2))*self.scale,dim=-1)
        out=torch.bmm(A,V).transpose(0,1).contiguous().view(N,-1)
        # Residual fusion + L2 normalisation
        F_fuse=self.W_o(out)+x_app
        return F.normalize(F_fuse,p=2,dim=-1)   # (N,C) F_norm


class Module1_ForensicExtraction(nn.Module):
    """
    Module 1: Forensic Feature Extraction + Cross-Attention.
      ResNet-50 → F_app
      SRM + CNN → F_srm
      Multi-head cross-attention (Q,K=SRM | V=App)
      Residual + L2 norm → F_norm
      Skip connections: S0(C/4), S1(C/2), S2(C)
    """
    def __init__(self,C=256,num_heads=4):
        super().__init__()
        # Appearance stream
        bb=models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.enc0=nn.Sequential(bb.conv1,bb.bn1,bb.relu)
        self.pool=bb.maxpool; self.enc1=bb.layer1; self.enc2=bb.layer2
        for m in [self.enc0,self.pool,self.enc1]:
            for p in m.parameters(): p.requires_grad=False
        self.proj_app=nn.Sequential(nn.Conv2d(512,C,1),nn.BatchNorm2d(C),nn.ReLU(inplace=True))
        # Forensic noise stream
        self.srm     =SRMLayer(out_ch=C//2)
        self.proj_srm=nn.Sequential(nn.Conv2d(C//2,C,1),nn.BatchNorm2d(C),nn.ReLU(inplace=True))
        # Multi-head forensic cross-attention
        self.attn    =MultiHeadForensicAttention(C,num_heads)
        # Skip connections
        self.skip0=nn.Sequential(nn.Conv2d(64,C//4,1),nn.BatchNorm2d(C//4),nn.ReLU(inplace=True))
        self.skip1=nn.Sequential(nn.Conv2d(256,C//2,1),nn.BatchNorm2d(C//2),nn.ReLU(inplace=True))
        self.skip2=nn.Sequential(nn.Conv2d(512,C,1),nn.BatchNorm2d(C),nn.ReLU(inplace=True))

    def forward(self,x):
        # Dual-stream extraction
        s0=self.enc0(x); s1=self.enc1(self.pool(s0)); s2=self.enc2(s1)
        f_app=self.proj_app(s2)          # (B,C,H/8,W/8) appearance
        f_srm=self.proj_srm(self.srm(x)) # (B,C,H/8,W/8) forensic noise
        f_app_32=f_app                    # save for decoder skip

        # Downsample to 16×16=256 nodes
        f_app_d=F.avg_pool2d(f_app,2)   # (B,C,16,16)
        f_srm_d=F.avg_pool2d(f_srm,2)   # (B,C,16,16)
        B,C,H,W=f_app_d.shape; N=H*W

        # Per-image cross-attention
        f_norm_list=[]
        x_srm_list =[]
        for b in range(B):
            xa=f_app_d[b].permute(1,2,0).reshape(N,C)
            xs=f_srm_d[b].permute(1,2,0).reshape(N,C)
            f_norm_list.append(self.attn(xa,xs))   # (N,C) L2-normalised
            x_srm_list.append(xs)

        skips=[self.skip0(s0),self.skip1(s1),self.skip2(s2)]
        return f_norm_list, x_srm_list, f_app_32, skips, (H,W)


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 2 — HyperGIN Block
# ═══════════════════════════════════════════════════════════════════════════

class HyperGINBlock(nn.Module):
    """
    HyperGIN Block — core novel module.

    Step 1: Adaptive 2-criteria Hypergraph
      h_feat = W_feat · F_norm   (feature similarity)
      h_srm  = W_srm  · X_srm   (SRM consistency)
      a      = softmax(v·tanh(W·F_norm))  (suspicion)
      H      = softmax(w1·h_feat + w2·h_srm)

    Step 2: Attention-weighted HGNN × 2
      F_hyp^(l) = H^T·(a⊙X^(l-1)) / D_e
      X^(l)     = LN(ReLU(Θ_l·H·F_hyp^(l)) + X^(l-1))

    Step 3: Displacement-aware GIN
      d*    = Σ A[n,m]·d_nm / Σ A[n,m]
      W_nm  = exp(-||d_nm - d*||²/2σ²),  σ learnable
      A_w   = A ⊙ W  (row-normalised)
      F_gin = LN(MLP((1+ε)·X + A_w·X))

    Step 4: Adaptive Forensic Gate
      g     = sigmoid(W_g·[X||GIN_msg||H_msg])
      F_out = LN(MLP(g⊙(X + GIN_msg + H_msg)) + X)
    """
    def __init__(self,C,K=16,top_k=10):
        super().__init__()
        # Step 1: Hypergraph
        self.W_feat=nn.Linear(C,K)
        self.W_srm =nn.Linear(C,K)
        self.w     =nn.Parameter(torch.ones(2)/2)
        self.W_sus =nn.Linear(C,C//4)
        self.v_sus =nn.Linear(C//4,1,bias=False)
        # Step 2: HGNN × 2
        self.theta1=nn.Linear(C,C,bias=False); self.norm1=nn.LayerNorm(C)
        self.theta2=nn.Linear(C,C,bias=False); self.norm2=nn.LayerNorm(C)
        # Step 3: GIN
        self.top_k =top_k
        self.sigma =nn.Parameter(torch.tensor(0.5))
        self.eps   =nn.Parameter(torch.zeros(1))
        self.gin_mlp=nn.Sequential(nn.Linear(C,C),nn.ReLU(inplace=True),nn.Linear(C,C))
        self.gin_norm=nn.LayerNorm(C)
        # Step 4: Gate + MLP
        self.W_gate=nn.Linear(C*3,C)
        self.mlp   =nn.Sequential(nn.Linear(C,C*2),nn.GELU(),nn.Linear(C*2,C))
        self.norm  =nn.LayerNorm(C)

    def _hypergraph(self,x,x_srm):
        sus  =torch.softmax(self.v_sus(torch.tanh(self.W_sus(x))),dim=0)
        w    =F.softmax(self.w,dim=0)
        H    =F.softmax(w[0]*self.W_feat(x)+w[1]*self.W_srm(x_srm),dim=1)
        De   =H.sum(0).clamp(min=1e-6)
        # HGNN layer 1
        Fh1  =(H.T@(sus*x))/De.unsqueeze(1)
        x1   =self.norm1(F.relu(self.theta1(H@Fh1))+x)
        # HGNN layer 2
        Fh2  =(H.T@(sus*x1))/De.unsqueeze(1)
        x2   =self.norm2(F.relu(self.theta2(H@Fh2))+x1)
        return x2, H@Fh2   # refined embeddings + group message

    def _forensic_gin(self,x,coords):
        N=x.size(0)
        Xn=F.normalize(x,p=2,dim=1); sim=Xn@Xn.T; sim.fill_diagonal_(-1e4)
        k=min(self.top_k,N-1); vals,idx=sim.topk(k,dim=1)
        A=torch.zeros(N,N,device=x.device)
        A.scatter_(1,idx,vals.clamp(min=0).float())
        A=(A+A.T)/2.0; A=A/A.sum(1,keepdim=True).clamp(min=1e-6)
        diff  =coords.unsqueeze(0)-coords.unsqueeze(1)
        wA    =A.unsqueeze(2)
        d_star=(wA*diff).sum(dim=(0,1))/(wA.sum()+1e-6)
        dist2 =((diff-d_star)**2).sum(dim=2)
        sigma =self.sigma.abs().clamp(min=0.05)
        W     =torch.exp(-dist2/(2*sigma**2))
        A_w   =A*W; A_w=A_w/A_w.sum(1,keepdim=True).clamp(min=1e-6)
        gin_msg=A_w@x
        F_gin =self.gin_norm(self.gin_mlp((1+self.eps)*x+gin_msg))
        return F_gin, gin_msg

    def forward(self,x,x_srm,coords):
        # Step 1+2: Hypergraph + HGNN
        x_hgnn,H_msg=self._hypergraph(x,x_srm)
        # Step 3: Forensic GIN
        x_gin,GIN_msg=self._forensic_gin(x_hgnn,coords)
        # Step 4: Adaptive forensic gate
        gate_in=torch.cat([x_hgnn,GIN_msg,H_msg],dim=-1)
        g      =torch.sigmoid(self.W_gate(gate_in))
        fused  =g*(x_hgnn+GIN_msg+H_msg)
        out    =self.mlp(fused)
        return self.norm(out+x_hgnn)   # (N,C)


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 3 — U-Net Decoder
# ═══════════════════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    def __init__(self,in_ch,skip_ch,out_ch):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch+skip_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
    def forward(self,x,skip=None):
        x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
        if skip is not None:
            if skip.shape[2:]!=x.shape[2:]:
                skip=F.interpolate(skip,size=x.shape[2:],mode='bilinear',align_corners=True)
            x=torch.cat([x,skip],dim=1)
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════════
#  FULL MODEL
# ═══════════════════════════════════════════════════════════════════════════

class HyperGIN(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        C =cfg['embed_dim']; K=cfg['num_hyperedges']
        h =cfg['num_heads']; tk=cfg['top_k']; nl=cfg['num_layers']

        self.module1=Module1_ForensicExtraction(C,h)
        self.module2=nn.ModuleList([HyperGINBlock(C,K,tk) for _ in range(nl)])
        self.dec3   =DecoderBlock(C,    C,    C//2)
        self.dec2   =DecoderBlock(C//2, C//2, C//4)
        self.dec1   =DecoderBlock(C//4, C//4, C//8)
        self.head   =nn.Conv2d(C//8,1,1)

    def _coords(self,H,W,device):
        ys=torch.linspace(0,1,H,device=device)
        xs=torch.linspace(0,1,W,device=device)
        gy,gx=torch.meshgrid(ys,xs,indexing='ij')
        return torch.stack([gx.flatten(),gy.flatten()],dim=1)

    def forward(self,x):
        B,_,H_in,W_in=x.shape
        # Module 1
        f_norm_list,x_srm_list,f_app_32,skips,(H,W)=self.module1(x)
        coords=self._coords(H,W,x.device)
        # Module 2
        out=[]
        for b in range(B):
            feat=f_norm_list[b]; xs=x_srm_list[b]
            for block in self.module2:
                feat=block(feat,xs,coords)
            out.append(feat.reshape(H,W,feat.size(-1)).permute(2,0,1))
        g=torch.stack(out,dim=0)   # (B,C,H,W)
        # Module 3
        C=g.size(1)
        d=self.dec3(g,    f_app_32)
        d=self.dec2(d, skips[1])
        d=self.dec1(d, skips[0])
        d=F.interpolate(d,size=(H_in,W_in),mode='bilinear',align_corners=True)
        return self.head(d)


# ═══════════════════════════════════════════════════════════════════════════
#  LOSS + METRICS + EVAL
# ═══════════════════════════════════════════════════════════════════════════

def dice_loss(p,t,eps=1e-6):
    inter=(p*t).sum(dim=(2,3))
    return (1-(2*inter+eps)/(p.sum(dim=(2,3))+t.sum(dim=(2,3))+eps)).mean()

def compute_loss(logits,masks,cfg):
    pw  =torch.tensor([cfg['pos_weight']],device=logits.device)
    bce =F.binary_cross_entropy_with_logits(logits,masks,pos_weight=pw)
    dice=dice_loss(torch.sigmoid(logits),masks)
    return bce+cfg['dice_w']*dice, bce.item(), dice.item()

def thr_sweep(ap,am,cfg):
    probs=np.concatenate([p.flatten() for p in ap])
    masks=np.concatenate([m.flatten() for m in am]).astype(bool)
    best={'f1':0.,'thr':0.5,'precision':0.,'recall':0.,'iou':0.}
    for thr in np.linspace(cfg['thr_min'],cfg['thr_max'],cfg['thr_steps']):
        pred=probs>=thr
        tp=(pred&masks).sum(); fp=(pred&~masks).sum(); fn=(~pred&masks).sum()
        pr=tp/(tp+fp+1e-8); rc=tp/(tp+fn+1e-8)
        f1=2*pr*rc/(pr+rc+1e-8); iou=tp/(tp+fp+fn+1e-8)
        if f1>best['f1']:
            best={'f1':float(f1),'thr':float(thr),'precision':float(pr),
                  'recall':float(rc),'iou':float(iou)}
    return best

@torch.no_grad()
def evaluate(model,loader,cfg,tau=0.5):
    model.eval(); ap,am,viz=[],[],[]
    for imgs,masks in loader:
        imgs=imgs.to(DEVICE)
        probs=torch.sigmoid(model(imgs)).cpu().numpy()
        for b in range(probs.shape[0]):
            p=probs[b,0]; m=masks[b,0].numpy().astype(bool)
            ap.append(p); am.append(m)
            if len(viz)<6:
                pred=(p>=tau).astype(bool)
                tp=(pred&m).sum(); fp=(pred&~m).sum(); fn=(~pred&m).sum()
                pr=tp/(tp+fp+1e-8); rc=tp/(tp+fn+1e-8)
                viz.append({'prob':p,'pred':pred,'gt':m,
                            'f1':float(2*pr*rc/(pr+rc+1e-8))})
    return thr_sweep(ap,am,cfg), viz


# ═══════════════════════════════════════════════════════════════════════════
#  SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════

def save_qual(viz,out_dir,tag):
    n=len(viz)
    if n==0: return
    fig,axes=plt.subplots(n,3,figsize=(10,3.5*n))
    if n==1: axes=[axes]
    for i,r in enumerate(viz):
        axes[i][0].imshow(r['prob'],cmap='hot',vmin=0,vmax=1); axes[i][0].axis('off')
        if i==0: axes[i][0].set_title('Prob Map',fontweight='bold')
        axes[i][1].imshow(r['gt'],cmap='gray'); axes[i][1].axis('off')
        if i==0: axes[i][1].set_title('Ground Truth',fontweight='bold')
        axes[i][2].imshow(r['pred'],cmap='gray'); axes[i][2].axis('off')
        if i==0: axes[i][2].set_title('HyperGIN',fontweight='bold')
        axes[i][0].set_ylabel(f"F1={r['f1']:.3f}",fontsize=9,
                              rotation=0,labelpad=55,va='center')
    plt.suptitle(f'HyperGIN — {tag}',fontsize=11,fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,f'{tag}.png'),dpi=150,bbox_inches='tight')
    plt.close(); print(f"[Output] {tag}.png")

def save_curves(history,out_dir):
    if not history: return
    e=[h['epoch'] for h in history]
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    ax[0].plot(e,[h['train_loss'] for h in history],'b-o',ms=3,lw=2)
    ax[0].set_title('Training Loss',fontsize=12,fontweight='bold')
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Loss'); ax[0].grid(True)
    ax[1].plot(e,[h['val_f1'] for h in history],'g-o',ms=3,lw=2,label='Val F1')
    ax[1].plot(e,[h['precision'] for h in history],'b--s',ms=3,lw=2,label='Precision')
    ax[1].plot(e,[h['recall'] for h in history],'r--^',ms=3,lw=2,label='Recall')
    ax[1].set_title('Pixel-Level Validation Metrics',fontsize=12,fontweight='bold')
    ax[1].set_xlabel('Epoch'); ax[1].legend(); ax[1].grid(True)
    plt.suptitle('HyperGIN — Training Progress',fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'training_curves.png'),dpi=150,bbox_inches='tight')
    plt.close(); print(f"[Output] training_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  HyperGIN — Final 3-Module Pipeline")
print(f"  Module 1: Forensic Feature Extraction + Cross-Attention")
print(f"    ResNet-50 + SRM + Multi-head Attn (h={CFG['num_heads']}) + L2 Norm")
print(f"  Module 2: HyperGIN Block × {CFG['num_layers']}  [NOVEL]")
print(f"    2-criteria Hypergraph + Attn-HGNN + Forensic GIN + Gate")
print(f"  Module 3: U-Net Decoder → pixel mask")
print(f"{'='*65}\n")

all_samples=parse_all_comofod(ROOT)
train_s,val_s,test_s=split_samples(all_samples,CFG)

print("\n[Dataset] Building:")
print("  Train:"); train_ds=FullImageDataset(train_s,CFG,augment=True)
print("  Val:");   val_ds  =FullImageDataset(val_s,  CFG,augment=False)
print("  Test:");  test_ds =FullImageDataset(test_s, CFG,augment=False)

kw=dict(num_workers=CFG['num_workers'],pin_memory=True)
tl=DataLoader(train_ds,batch_size=CFG['batch_size'],shuffle=True, **kw)
vl=DataLoader(val_ds,  batch_size=CFG['batch_size'],shuffle=False,**kw)
sl=DataLoader(test_ds, batch_size=CFG['batch_size'],shuffle=False,**kw)

model=HyperGIN(CFG).to(DEVICE)
total=sum(p.numel() for p in model.parameters())
trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[Model] total={total:,} | trainable={trainable:,}")

optimizer=optim.AdamW([
    {'params':model.module1.parameters(),  'lr':CFG['lr']*0.1},
    {'params':model.module2.parameters(),  'lr':CFG['lr']},
    {'params':model.dec3.parameters(),     'lr':CFG['lr']},
    {'params':model.dec2.parameters(),     'lr':CFG['lr']},
    {'params':model.dec1.parameters(),     'lr':CFG['lr']},
    {'params':model.head.parameters(),     'lr':CFG['lr']},
],weight_decay=CFG['weight_decay'])
scheduler=optim.lr_scheduler.CosineAnnealingLR(
    optimizer,T_max=CFG['epochs'],eta_min=1e-7)

start=1; best_f1=0.; best_ep=1; best_thr=0.5; pat=0; history=[]
if os.path.exists(TRAIN_CKPT):
    ck=torch.load(TRAIN_CKPT,map_location=DEVICE,weights_only=False)
    model.load_state_dict(ck['model_state'])
    optimizer.load_state_dict(ck['optimizer_state'])
    scheduler.load_state_dict(ck['scheduler_state'])
    start=ck['epoch']+1; best_f1=ck['best_f1']
    best_ep=ck['best_ep']; best_thr=ck['best_thr']
    pat=ck['pat']; history=ck.get('history',[])
    print(f"[Train] Resumed epoch {start-1}, best F1={best_f1:.4f}")

print(f"\n{'Ep':>4} | {'Loss':>7} | {'BCE':>6} | {'Dice':>5} | "
      f"{'ValF1':>7} | {'P':>6} | {'R':>6} | {'Thr':>5} | {'Time':>6}")
print(f"{'─'*75}")

for ep in range(start,CFG['epochs']+1):
    t0=time.time(); model.train(); trl=bcel=dicel=0.; ns=0
    for imgs,masks in tl:
        imgs=imgs.to(DEVICE,non_blocking=True)
        masks=masks.to(DEVICE,non_blocking=True)
        optimizer.zero_grad()
        logits=model(imgs)
        loss,bce,dice=compute_loss(logits,masks,CFG)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        trl+=loss.item(); bcel+=bce; dicel+=dice; ns+=1
    scheduler.step()
    n=max(1,ns); trl/=n; bcel/=n; dicel/=n
    m,viz=evaluate(model,vl,CFG,tau=best_thr)
    best_thr=m['thr']; elapsed=time.time()-t0

    print(f"{ep:>4} | {trl:>7.4f} | {bcel:>6.3f} | {dicel:>5.3f} | "
          f"{m['f1']:>7.4f} | {m['precision']:>6.4f} | {m['recall']:>6.4f} | "
          f"{m['thr']:>5.2f} | {elapsed:>5.1f}s")

    history.append({'epoch':ep,'train_loss':trl,'val_f1':m['f1'],
                    'precision':m['precision'],'recall':m['recall'],'iou':m['iou']})

    if m['f1']>best_f1:
        best_f1=m['f1']; best_ep=ep
        torch.save({'epoch':ep,'model_state':model.state_dict(),
                    'val_f1':m['f1'],'precision':m['precision'],
                    'recall':m['recall'],'iou':m['iou'],
                    'threshold':m['thr'],'cfg':CFG},MODEL_PT)
        print(f"       ✅ Best F1={m['f1']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} IoU={m['iou']:.4f} τ={m['thr']:.2f}")
        save_qual(viz,QUAL_DIR,f'val_ep{ep}'); pat=0
    else: pat+=1

    torch.save({'epoch':ep,'model_state':model.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'scheduler_state':scheduler.state_dict(),
                'best_f1':best_f1,'best_ep':best_ep,'best_thr':best_thr,
                'pat':pat,'history':history,'cfg':CFG},TRAIN_CKPT)

    if pat>=CFG['patience']:
        print(f"\n[Train] Early stop at epoch {ep}"); break

print(f"\n[Test] Loading best model (epoch={best_ep})...")
ck=torch.load(MODEL_PT,map_location=DEVICE,weights_only=False)
model.load_state_dict(ck['model_state'])
test_m,test_viz=evaluate(model,sl,CFG,tau=ck['threshold'])

save_curves(history,OUT_DIR)
save_qual(test_viz,QUAL_DIR,'test_final')

lines=["="*62,"  HyperGIN — Final Test Results",
       "  Module 1: Forensic Extraction + Multi-head Cross-Attention",
       "  Module 2: HyperGIN Block (Hypergraph+GIN+Gate) × 2",
       "  Module 3: U-Net Decoder",
       "="*62,f"  Best epoch : {best_ep}",
       f"  {'─'*50}",f"  {'Metric':<28}{'Value':>10}",f"  {'─'*50}",
       f"  {'F1':<28}{test_m['f1']:>10.4f}",
       f"  {'Precision':<28}{test_m['precision']:>10.4f}",
       f"  {'Recall':<28}{test_m['recall']:>10.4f}",
       f"  {'IoU':<28}{test_m['iou']:>10.4f}",
       f"  {'Threshold':<28}{test_m['thr']:>10.2f}",
       f"  {'─'*50}","="*62]
txt='\n'.join(lines)
with open(os.path.join(OUT_DIR,'final_results.txt'),'w') as f: f.write(txt+'\n')
print(f"\n{txt}")

if history:
    p=os.path.join(OUT_DIR,'training_history.csv')
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys())
        w.writeheader(); w.writerows(history)

print(f"\n✅ HyperGIN training complete! → {OUT_DIR}")
