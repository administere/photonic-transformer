#!/usr/bin/env python3
"""
光子 Attention 点积验证 · v8 完整工程验证
一次性输出所有关键指标
"""
import numpy as np
from scipy.stats import spearmanr, entropy
from scipy.special import softmax
import time

N_PAIRS = 500; N_BOOTSTRAP = 20
DEFAULT_DAC = 4; DEFAULT_CALIB = 0.02; DEFAULT_POWER = 2.0
np.random.seed(42)

def mzi_T(phi): return np.sin(phi/2.0)**2
def predistort_unipolar(q): return 2.0*np.arcsin(np.sqrt(np.clip(q,1e-15,1-1e-15)))
def predistort_bipolar(q):
    w = 2.0*q-1.0; return np.arcsin(np.clip(w,-1+1e-12,1-1e-12))
def apply_dac(phi,lo,hi,bits):
    levels=2**bits; delta=(hi-lo)/(levels-1)
    q=np.round((phi-lo)/delta)*delta+lo; return np.clip(q,lo,hi)
def apply_calib_error(phi,err,spatial_corr=False):
    if spatial_corr:
        g=np.random.normal(0,err*0.5); n=np.random.normal(0,err*0.5,phi.shape)
        noise=g+n
    else: noise=np.random.normal(0,err,phi.shape)
    return phi*(1.0+noise)+np.random.normal(0,err*0.05,phi.shape)
def detector_noise(I,p_ch,saturate=False):
    if saturate: I=np.tanh(I/(0.5*p_ch))*p_ch
    I_abs=np.abs(I)+1e-15
    shot=np.random.randn(*I.shape)*np.sqrt(I_abs)*0.008
    thermal=np.random.randn(*I.shape)*0.003
    rin=np.random.randn(*I.shape)*I_abs*0.0005
    return I+shot+thermal+rin

def photonic_dot(q,k,cfg):
    bipolar=cfg['bipolar']; dpm=cfg['delta_phi_max']; pwr=cfg['power_mW']
    dac=cfg['dac_bits']; cerr=cfg['calib_error']
    fanout=cfg.get('fanout_loss',True); wdm=cfg.get('wdm_mode',False)
    spc=cfg.get('spatial_corr',False); sat=cfg.get('saturate',False)
    D=len(q); p_ch=pwr if wdm else (pwr/D if fanout else pwr)
    if bipolar:
        phi=predistort_bipolar(q)*(dpm/(np.pi/2)); phi=apply_calib_error(phi,cerr,spc)
        phi=apply_dac(phi,-dpm,dpm,dac)
        Tpp=mzi_T(np.pi/2+phi); Tpn=mzi_T(np.pi/2-phi)
        Tnp=mzi_T(-np.pi/2+phi); Tnn=mzi_T(-np.pi/2-phi)
        Ipp=detector_noise(Tpp*p_ch*k,p_ch,sat); Ipn=detector_noise(Tpn*p_ch*k,p_ch,sat)
        Inp=detector_noise(Tnp*p_ch*k,p_ch,sat); Inn=detector_noise(Tnn*p_ch*k,p_ch,sat)
        raw=np.sum((Ipp+Inn)-(Ipn+Inp)); offset=2.0*p_ch*np.sum(k)
        return (raw+offset)/(4.0*p_ch)
    else:
        phi=predistort_unipolar(q)*(dpm/np.pi); phi=apply_calib_error(phi,cerr,spc)
        phi=apply_dac(phi,0,dpm,dac)
        T=mzi_T(phi); I=detector_noise(T*p_ch*k,p_ch,sat)
        dot_raw=np.sum(I); nf=D*mzi_T(dpm)*p_ch
        return dot_raw/nf*(D*p_ch)

def generate_realistic_embeddings(n,dim):
    x=np.random.lognormal(mean=0.0,sigma=0.5,size=(n,dim))
    mask=np.random.rand(n,dim)<0.1; x[mask]=np.random.exponential(scale=2.0,size=mask.sum())
    row_max=np.max(x,axis=1,keepdims=True); row_max[row_max==0]=1.0
    x=x/row_max; return np.clip(x,1e-12,1-1e-12)

def evaluate(cfg,seq_len,n_pairs,n_boot,realistic=False):
    rho_vals=[]; kl_vals=[]
    for _ in range(n_boot):
        if realistic: q=generate_realistic_embeddings(n_pairs,seq_len); k=generate_realistic_embeddings(n_pairs,seq_len)
        else: q=np.random.rand(n_pairs,seq_len).astype(np.float32); k=np.random.rand(n_pairs,seq_len).astype(np.float32)
        ideal=np.sum(q*k,axis=1)
        phot=np.array([photonic_dot(q[i],k[i],cfg) for i in range(n_pairs)])
        mask=np.isfinite(ideal)&np.isfinite(phot)
        if mask.sum()<2: continue
        rho,_=spearmanr(ideal[mask],phot[mask])
        if not np.isnan(rho): rho_vals.append(rho)
        att_ideal=softmax(ideal[mask]); att_phot=softmax(phot[mask])
        kl=entropy(att_ideal+1e-12,att_phot+1e-12); kl_vals.append(kl)
    if not rho_vals: return 0,0,0,0
    return np.mean(rho_vals),np.std(rho_vals),np.mean(kl_vals),np.std(kl_vals)

def matrix_multiply_sim(Q,K,cfg):
    nq,d=Q.shape; nk=K.shape[0]; S=np.zeros((nq,nk))
    for i in range(nq):
        for j in range(nk): S[i,j]=photonic_dot(Q[i],K[j],cfg)
    return S

# ========================= 主验证 =========================
print("="*70)
print("  光子 Attention 点积验证 · v8 完整工程分析")
print("="*70)
t0=time.time()

cfg_uni=dict(bipolar=False,delta_phi_max=np.pi,power_mW=DEFAULT_POWER,dac_bits=DEFAULT_DAC,
             calib_error=DEFAULT_CALIB,fanout_loss=True,wdm_mode=False,spatial_corr=False,saturate=False)
cfg_bip=dict(bipolar=True,delta_phi_max=np.pi/2,power_mW=DEFAULT_POWER,dac_bits=DEFAULT_DAC,
             calib_error=DEFAULT_CALIB,fanout_loss=True,wdm_mode=False,spatial_corr=False,saturate=False)
cfg_bip_wdm=dict(bipolar=True,delta_phi_max=np.pi/2,power_mW=DEFAULT_POWER,dac_bits=DEFAULT_DAC,
                 calib_error=DEFAULT_CALIB,fanout_loss=False,wdm_mode=True,spatial_corr=False,saturate=False)

print("\n▶ 1. 大维度缩放")
for dim in [64,128]:
    cfg=dict(bipolar=False,delta_phi_max=np.pi,power_mW=DEFAULT_POWER,dac_bits=DEFAULT_DAC,
             calib_error=DEFAULT_CALIB,fanout_loss=True,wdm_mode=False,spatial_corr=False,saturate=False)
    r,s,kl,_=evaluate(cfg,dim,N_PAIRS//2,N_BOOTSTRAP,realistic=False)
    print(f"  D={dim:3d}  ρ={r:.4f}±{s:.4f}  KL={kl:.4f}  {'✅' if r>=0.99 else '⚠️'}")

print("\n▶ 2. DAC 位数下限")
for b in [2,3,4]:
    cfg=dict(bipolar=False,delta_phi_max=np.pi,power_mW=DEFAULT_POWER,dac_bits=b,
             calib_error=DEFAULT_CALIB,fanout_loss=True,wdm_mode=False,spatial_corr=False,saturate=False)
    r,s,kl,_=evaluate(cfg,64,N_PAIRS//2,N_BOOTSTRAP,realistic=False)
    print(f"  {b}-bit  ρ={r:.4f}±{s:.4f}  KL={kl:.4f}  {'✅' if r>=0.99 else '⚠️'}")

print("\n▶ 3. 校准误差容忍度")
for e in [0.02,0.05,0.10,0.15]:
    cfg=dict(bipolar=False,delta_phi_max=np.pi,power_mW=DEFAULT_POWER,dac_bits=4,
             calib_error=e,fanout_loss=True,wdm_mode=False,spatial_corr=False,saturate=False)
    r,s,kl,_=evaluate(cfg,64,N_PAIRS//2,N_BOOTSTRAP,realistic=False)
    print(f"  err={e*100:2.0f}%  ρ={r:.4f}±{s:.4f}  KL={kl:.4f}  {'✅' if r>=0.99 else '⚠️'}")

print("\n▶ 4. softmax 保真度 (KL)")
for name,cfg in [('UNI',cfg_uni),('BIP',cfg_bip),('BIP_WDM',cfg_bip_wdm)]:
    r,s,kl,ks=evaluate(cfg,8,N_PAIRS,N_BOOTSTRAP,realistic=False)
    print(f"  {name:10s}  ρ={r:.4f}±{s:.4f}  KL={kl:.4f}±{ks:.4f}")

print("\n▶ 5. 矩阵乘法 (MVM) 8×8")
Q=np.random.rand(8,8).astype(np.float32); K=np.random.rand(8,8).astype(np.float32)
Si=Q@K.T; Sp=matrix_multiply_sim(Q,K,cfg_uni)
cos=np.sum(Si*Sp)/(np.linalg.norm(Si)*np.linalg.norm(Sp)+1e-15)
rel=np.mean(np.abs(Si-Sp)/(np.abs(Si)+1e-12))
print(f"  余弦相似度: {cos:.6f}  平均相对误差: {rel:.4f}")

print("\n▶ 6. 探测器饱和")
cs=cfg_uni.copy(); cs['saturate']=True
cn=cfg_uni.copy(); cn['saturate']=False
rs,_,kls,_=evaluate(cs,8,N_PAIRS,N_BOOTSTRAP,realistic=False)
rn,_,kln,_=evaluate(cn,8,N_PAIRS,N_BOOTSTRAP,realistic=False)
print(f"  无饱和: ρ={rn:.4f} KL={kln:.4f}  有饱和: ρ={rs:.4f} KL={kls:.4f}  {'✅' if rs>=0.99 else '⚠️'}")

print("\n▶ 7. 空间相关误差")
cc=dict(bipolar=False,delta_phi_max=np.pi,power_mW=DEFAULT_POWER,dac_bits=4,
        calib_error=0.05,fanout_loss=True,wdm_mode=False,spatial_corr=True,saturate=False)
ci=dict(bipolar=False,delta_phi_max=np.pi,power_mW=DEFAULT_POWER,dac_bits=4,
        calib_error=0.05,fanout_loss=True,wdm_mode=False,spatial_corr=False,saturate=False)
rc,_,klc,_=evaluate(cc,8,N_PAIRS,N_BOOTSTRAP,realistic=False)
ri,_,kli,_=evaluate(ci,8,N_PAIRS,N_BOOTSTRAP,realistic=False)
print(f"  独立: ρ={ri:.4f} KL={kli:.4f}  空间相关: ρ={rc:.4f} KL={klc:.4f}  {'✅' if rc>=0.99 else '⚠️'}")

print("\n"+"="*70)
print(f"总耗时: {time.time()-t0:.1f}s")
print("="*70)
