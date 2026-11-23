"""
V16-B: Softplus ANN with PDE + boundary-condition regularisation.
Run from project root: python train_ann_v16_physics_regularized.py

Required dataset files:
  microchannel_dataset.csv
  microchannel_physics_features.csv
  ann_v12_validation_case_split.csv

V16-B uses the V13 nine features, sqrt(Temperature) target, 128-64 network,
same saved LOPO/inner-validation splits, and Softplus activation for smooth
second derivatives. Outputs use ann_v16_ prefixes; V13 files are untouched.
"""
import os, time, copy, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
MAX_EPOCHS = 200
BATCH_SIZE = 512
LR, MIN_LR = 1e-3, 1e-5
EARLY_PATIENCE, LR_PATIENCE, MIN_DELTA = 20, 7, 1e-5
LAMBDA_PDE = 0.01
LAMBDA_BC = 0.01
N_PDE = 512
N_BC = 128
DATA_DIR = "dataset"
DEVICE_OVERRIDE = "auto"  # auto, cpu, cuda

DEVICE = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE_OVERRIDE == "auto" else DEVICE_OVERRIDE
)
if DEVICE.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but unavailable.")
if DEVICE.type == "cpu":
    torch.set_num_threads(min(8, os.cpu_count() or 1))
os.makedirs(DATA_DIR, exist_ok=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y, p):
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "R2": r2_score(y, p),
        "Max_Error": np.max(np.abs(y-p))
    }


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128), nn.Softplus(beta=1.0),
            nn.Linear(128, 64), nn.Softplus(beta=1.0),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def physical_features(x, y, pe, kn, tmac):
    """Differentiable recreation of all nine V13 inputs."""
    beta = (2.0-tmac)/tmac
    u = 6.0*(y*(1.0-y)+kn*beta)/(1.0+6.0*kn*beta)
    pe_u = pe*u
    dx_hot = torch.maximum(
        torch.maximum(4.0-x, torch.zeros_like(x)), x-6.0
    )
    d_hot = torch.sqrt(dx_hot.square()+y.square()+1e-12)
    heated = ((x >= 0.0) & (x <= 10.0)).to(x.dtype)
    return torch.cat([
        x, y, torch.log10(pe), torch.log10(1.0+kn), tmac,
        u, pe_u, d_hot, heated
    ], dim=1)


def theta_prediction(model, raw, xm, xs, ym, ys):
    scaled_sqrt = model((raw-xm)/xs)
    sqrt_theta = scaled_sqrt*ys+ym
    return sqrt_theta.square()


def physics_loss(model, cases, xm, xs, ym, ys, theta_ref):
    """
    PDE residual: theta_XX + theta_YY - Pe*u*theta_X = 0.
    BCs: inlet theta=0; outlet theta_X=0; top theta_Y=0;
    bottom theta_Y=-q(X), q=1 on [0,10], q=6 on [4,6].
    Cases are sampled exclusively from this fold's inner-training cases.
    """
    dev = xm.device
    # Interior collocation points
    ids = torch.randint(len(cases), (N_PDE,), device=dev)
    p = cases[ids]
    pe, kn, tm = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    x = (torch.rand(N_PDE, 1, device=dev)*30.0-10.0).requires_grad_(True)
    y = torch.rand(N_PDE, 1, device=dev).requires_grad_(True)
    th = theta_prediction(
        model, physical_features(x, y, pe, kn, tm), xm, xs, ym, ys
    )
    tx, ty = torch.autograd.grad(
        th.sum(), (x, y), create_graph=True, retain_graph=True
    )
    txx = torch.autograd.grad(
        tx.sum(), x, create_graph=True, retain_graph=True
    )[0]
    tyy = torch.autograd.grad(
        ty.sum(), y, create_graph=True, retain_graph=True
    )[0]
    beta = (2.0-tm)/tm
    u = 6.0*(y*(1.0-y)+kn*beta)/(1.0+6.0*kn*beta)
    residual = txx+tyy-pe*u*tx
    lpde = (residual/theta_ref).square().mean()

    def draw(n=N_BC):
        pp = cases[torch.randint(len(cases), (n,), device=dev)]
        return pp[:, 0:1], pp[:, 1:2], pp[:, 2:3]

    # Inlet X=-10: theta=0
    peb, knb, tmb = draw()
    xb = torch.full((N_BC,1), -10.0, device=dev)
    yb = torch.rand(N_BC,1, device=dev)
    xb.requires_grad_(True); yb.requires_grad_(True)
    tb = theta_prediction(
        model, physical_features(xb,yb,peb,knb,tmb),xm,xs,ym,ys
    )
    lin = (tb/theta_ref).square().mean()

    # Outlet X=20: theta_X=0
    peb,knb,tmb = draw()
    xb = torch.full((N_BC,1),20.0,device=dev).requires_grad_(True)
    yb = torch.rand(N_BC,1,device=dev).requires_grad_(True)
    tb = theta_prediction(
        model,physical_features(xb,yb,peb,knb,tmb),xm,xs,ym,ys
    )
    tbx = torch.autograd.grad(tb.sum(),xb,create_graph=True)[0]
    lout = ((30.0*tbx)/theta_ref).square().mean()

    # Top Y=1: theta_Y=0
    peb,knb,tmb = draw()
    xb = (torch.rand(N_BC,1,device=dev)*30.0-10.0).requires_grad_(True)
    yb = torch.ones(N_BC,1,device=dev).requires_grad_(True)
    tb = theta_prediction(
        model,physical_features(xb,yb,peb,knb,tmb),xm,xs,ym,ys
    )
    tby = torch.autograd.grad(tb.sum(),yb,create_graph=True)[0]
    ltop = (tby/theta_ref).square().mean()

    # Bottom Y=0: theta_Y=-q(X); hotspot flux is 6, not 5.
    peb,knb,tmb = draw()
    xb = (torch.rand(N_BC,1,device=dev)*30.0-10.0).requires_grad_(True)
    yb = torch.zeros(N_BC,1,device=dev).requires_grad_(True)
    tb = theta_prediction(
        model,physical_features(xb,yb,peb,knb,tmb),xm,xs,ym,ys
    )
    tby = torch.autograd.grad(tb.sum(),yb,create_graph=True)[0]
    q = torch.where((xb>=0.0)&(xb<=10.0),torch.ones_like(xb),torch.zeros_like(xb))
    q = torch.where((xb>=4.0)&(xb<=6.0),6.0*torch.ones_like(xb),q)
    lbot = ((tby+q)/theta_ref).square().mean()

    lbc = lin+lout+ltop+lbot
    details = {
        "BC_Inlet": lin.detach().item(),
        "BC_Outlet": lout.detach().item(),
        "BC_Top": ltop.detach().item(),
        "BC_Bottom": lbot.detach().item()
    }
    return lpde,lbc,details


def train_fold(Xtr,ytr,Xv,yv,cases_np,xsc,ysc,theta_ref,fold,heldout):
    seed_all(SEED+fold)
    xt=torch.tensor(Xtr,device=DEVICE); yt=torch.tensor(ytr,device=DEVICE)
    xv=torch.tensor(Xv,device=DEVICE); yv=torch.tensor(yv,device=DEVICE)
    xm=torch.tensor(xsc.mean_,dtype=torch.float32,device=DEVICE)
    xs=torch.tensor(xsc.scale_,dtype=torch.float32,device=DEVICE)
    ym=torch.tensor(float(ysc.mean_[0]),dtype=torch.float32,device=DEVICE)
    ys=torch.tensor(float(ysc.scale_[0]),dtype=torch.float32,device=DEVICE)
    tref=torch.tensor(max(float(theta_ref),1.0),dtype=torch.float32,device=DEVICE)
    cases=torch.tensor(cases_np,dtype=torch.float32,device=DEVICE)

    model=ANN().to(DEVICE)
    data_generator = torch.Generator(device=DEVICE)
    data_generator.manual_seed(SEED + fold + 10000)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,mode="min",factor=0.5,patience=LR_PATIENCE,min_lr=MIN_LR
    )
    best=float("inf"); best_epoch=0; best_state=None; wait=0
    history=[]; start=time.perf_counter()

    for epoch in range(1,MAX_EPOCHS+1):
        model.train()
        order=torch.randperm(len(xt),device=DEVICE,generator=data_generator)
        data_sum=0.0
        for s in range(0,len(xt),BATCH_SIZE):
            ids=order[s:s+BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(model(xt[ids]),yt[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            data_sum += loss.detach().item()*len(ids)
        data_mean=data_sum/len(xt)

        # One separate physics update per epoch keeps second-derivative
        # autograd cost manageable while applying PDE and BC regularisation.
        opt.zero_grad(set_to_none=True)
        lpde,lbc,bc_parts=physics_loss(model,cases,xm,xs,ym,ys,tref)
        weighted=LAMBDA_PDE*lpde+LAMBDA_BC*lbc
        weighted.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            val=nn.functional.mse_loss(model(xv),yv).item()
        scheduler.step(val)
        if val < best-MIN_DELTA:
            best=val; best_epoch=epoch
            best_state=copy.deepcopy(model.state_dict()); wait=0
        else:
            wait+=1
        history.append({
            "Fold":fold,"Heldout_Pe":heldout,"Epoch":epoch,
            "Train_Data_MSE_Sqrt_Scaled":data_mean,
            "PDE_Loss_Normalized":lpde.detach().item(),
            "BC_Loss_Normalized":lbc.detach().item(),
            "Weighted_Physics_Loss":weighted.detach().item(),
            **bc_parts,"Validation_MSE_Sqrt_Scaled":val,
            "Learning_Rate":opt.param_groups[0]["lr"],"Best_Epoch":best_epoch
        })
        if epoch==1 or epoch%10==0:
            print(f" epoch {epoch:3d} | data={data_mean:.6f} "
                  f"| PDE={lpde.item():.3e} | BC={lbc.item():.3e} "
                  f"| val={val:.6f} | best={best_epoch}")
        if wait>=EARLY_PATIENCE:
            print(f" Early stop at {epoch}; best epoch {best_epoch}")
            break
    if best_state is None:
        raise RuntimeError("No best checkpoint.")
    model.load_state_dict(best_state)
    return model,best_epoch,best,time.perf_counter()-start,history


# -------------------- Load data and reproduce saved V12 splits --------------------
seed_all(SEED)
raw=pd.read_csv(os.path.join(DATA_DIR,"microchannel_dataset.csv"))
feat=pd.read_csv(os.path.join(DATA_DIR,"microchannel_physics_features.csv"))
splits=pd.read_csv(os.path.join(DATA_DIR,"ann_v12_validation_case_split.csv"))
if len(raw)!=len(feat):
    raise ValueError("Raw and feature row counts differ.")
for c in ["X","Y","Pe","Kn","TMAC"]:
    if not np.allclose(raw[c],feat[c],equal_nan=True):
        raise ValueError(f"Row alignment failure: {c}")
case_df=raw[["Pe","Kn","TMAC"]].drop_duplicates()
if len(case_df)!=32:
    raise ValueError(f"Expected 32 CFD cases; found {len(case_df)}")

X=np.column_stack([
    raw["X"],raw["Y"],np.log10(raw["Pe"]),np.log10(1+raw["Kn"]),
    raw["TMAC"],feat["u_local"],feat["Pe_u"],feat["d_hotspot"],
    feat["heated_indicator"]
]).astype(np.float32)
T=raw["Temperature"].to_numpy(dtype=np.float32)
if np.any(T<0) or not np.isfinite(T).all():
    raise ValueError("Temperature must be finite and nonnegative.")
sqrtT=np.sqrt(T).astype(np.float32)
xp=raw["X"].to_numpy(); yp=raw["Y"].to_numpy()
pe=raw["Pe"].to_numpy(); kn=raw["Kn"].to_numpy(); tm=raw["TMAC"].to_numpy()
pe_values=sorted(np.unique(pe))
if len(pe_values)!=5:
    raise ValueError(f"Expected five Pe groups; got {pe_values}")
required={"Fold","Pe","Kn","TMAC","Split"}
if not required.issubset(splits.columns):
    raise ValueError("V12 split file has unexpected columns.")

all_cases=set(tuple(map(float,r)) for r in case_df.to_numpy())
for fold,held in enumerate(pe_values,1):
    fs=splits[splits["Fold"]==fold]
    keys=set(tuple(map(float,r)) for r in fs[["Pe","Kn","TMAC"]].drop_duplicates().to_numpy())
    expected={k for k in all_cases if k[0]!=float(held)}
    if keys!=expected:
        raise ValueError(f"V12 split case coverage mismatch in fold {fold}")

print("="*76)
print("V16-B | Softplus + PDE/BC | Same V12 LOPO splits")
print("Device:",DEVICE,"| Cases:",len(case_df),"| Rows:",len(raw))
print("lambda_PDE:",LAMBDA_PDE,"| lambda_BC:",LAMBDA_BC)
print("="*76)

pred=np.full(len(raw),np.nan,dtype=np.float32)
fold_results=[]; histories=[]; split_checks=[]
total_start=time.perf_counter()

for fold,held in enumerate(pe_values,1):
    outer=np.flatnonzero(pe!=held); test=np.flatnonzero(pe==held)
    fs=splits[splits["Fold"]==fold]
    val_keys=set(tuple(map(float,r)) for r in fs.loc[
        fs["Split"].str.lower()=="validation",["Pe","Kn","TMAC"]
    ].to_numpy())
    train_keys=set(tuple(map(float,r)) for r in fs.loc[
        fs["Split"].str.lower()=="training",["Pe","Kn","TMAC"]
    ].to_numpy())
    tr=outer[np.array([(float(pe[i]),float(kn[i]),float(tm[i])) in train_keys for i in outer])]
    va=outer[np.array([(float(pe[i]),float(kn[i]),float(tm[i])) in val_keys for i in outer])]
    if len(tr)==0 or len(va)==0:
        raise RuntimeError(f"Empty split in fold {fold}")

    split_checks.append({
        "Fold":fold,"Heldout_Pe":held,"Train_Points":len(tr),
        "Validation_Points":len(va),"Test_Points":len(test),
        "Train_Cases":len(train_keys),"Validation_Cases":len(val_keys)
    })
    xsc=StandardScaler(); ysc=StandardScaler()
    Xtr=xsc.fit_transform(X[tr]).astype(np.float32)
    Xv=xsc.transform(X[va]).astype(np.float32)
    Xtest=xsc.transform(X[test]).astype(np.float32)
    ytr=ysc.fit_transform(sqrtT[tr,None]).astype(np.float32)
    yv=ysc.transform(sqrtT[va,None]).astype(np.float32)
    # Crucial: physics boundary/collocation parameters are inner-training cases only.
    cases=raw.loc[tr,["Pe","Kn","TMAC"]].drop_duplicates().to_numpy(np.float32)
    print(f"\nFOLD {fold}/5 | held-out Pe={held} | train={len(tr)} val={len(va)} test={len(test)}")
    model,bepoch,bval,secs,hist=train_fold(
        Xtr,ytr,Xv,yv,cases,xsc,ysc,float(T[tr].max()),fold,held
    )
    histories.extend(hist)
    model.eval()
    with torch.no_grad():
        scaled=model(torch.tensor(Xtest,device=DEVICE)).cpu().numpy()
    pred_sqrt=ysc.inverse_transform(scaled).ravel()
    pred[test]=np.square(np.maximum(pred_sqrt,0.0))
    fm=metrics(T[test],pred[test])
    fold_results.append({
        "Model":"V16B_Softplus_Physics","Fold":fold,"Heldout_Pe":held,
        "Best_Epoch":bepoch,"Best_Validation_MSE":bval,
        "Runtime_Seconds":secs,**fm
    })
    print(f"V16-B | best epoch={bepoch} | MAE={fm['MAE']:.6f} "
          f"| RMSE={fm['RMSE']:.6f} | R2={fm['R2']:.6f}")
    del model
    if DEVICE.type=="cuda":
        torch.cuda.empty_cache()

if not np.isfinite(pred).all():
    raise RuntimeError("Missing or invalid out-of-fold predictions.")

global_df=pd.DataFrame([{
    "Model":"V16B_Softplus_Physics","Target":"sqrt(Temperature)",
    "Features":9,"Architecture":"128-64","Activation":"Softplus",
    "Lambda_PDE":LAMBDA_PDE,"Lambda_BC":LAMBDA_BC,**metrics(T,pred)
}])
per_pe=[]
for value in pe_values:
    m=pe==value
    per_pe.append({"Heldout_Pe":value,"Points":int(m.sum()),**metrics(T[m],pred[m])})
per_pe_df=pd.DataFrame(per_pe)

regions={
    "Entire domain":np.ones(len(T),dtype=bool),
    "Heated section":(xp>=0)&(xp<=10),
    "Hotspot streamwise band":(xp>=4)&(xp<=6),
    "Near-wall hotspot":(xp>=4)&(xp<=6)&(yp<=0.2)
}
regional=[]
for name,m in regions.items():
    regional.append({"Region":name,"Points":int(m.sum()),**metrics(T[m],pred[m])})
regional_df=pd.DataFrame(regional)

peak=[]
for c in case_df.itertuples(index=False):
    m=(pe==c.Pe)&(kn==c.Kn)&(tm==c.TMAC)
    actual=float(T[m].max()); estimate=float(pred[m].max())
    err=estimate-actual
    peak.append({
        "Pe":c.Pe,"Kn":c.Kn,"TMAC":c.TMAC,
        "Actual_Peak":actual,"Predicted_Peak":estimate,
        "Peak_Error":err,"Peak_Absolute_Error":abs(err),
        "Peak_Relative_Error_Pct":100*abs(err)/max(abs(actual),1e-12),
        "Direction":"over" if err>0 else "under" if err<0 else "exact"
    })
peak_df=pd.DataFrame(peak)

prediction_df=pd.DataFrame({
    "X":xp,"Y":yp,"Pe":pe,"Kn":kn,"TMAC":tm,
    "Actual_Temperature":T,"V16_Prediction":pred,
    "V16_Absolute_Error":np.abs(T-pred)
})
outputs={
    "oof_predictions":prediction_df,
    "global_results":global_df,
    "fold_results":pd.DataFrame(fold_results),
    "per_pe_results":per_pe_df,
    "regional_results":regional_df,
    "peak_temperature_results":peak_df,
    "training_history":pd.DataFrame(histories),
    "validation_split_check":pd.DataFrame(split_checks)
}
for suffix,df in outputs.items():
    df.to_csv(os.path.join(DATA_DIR,f"ann_v16b_{suffix}.csv"),index=False)

print("\nV16-B GLOBAL RESULTS")
print(global_df.to_string(index=False,float_format=lambda v:f"{v:.6f}"))
print("\nV16-B PER-PE RESULTS")
print(per_pe_df.to_string(index=False,float_format=lambda v:f"{v:.6f}"))
print("\nV16-B REGIONAL RESULTS")
print(regional_df.to_string(index=False,float_format=lambda v:f"{v:.6f}"))
print(f"\nTotal runtime: {(time.perf_counter()-total_start)/60:.2f} minutes")
print("Saved V16-B outputs only. V13 files were not modified.")
