from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd, torch, yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.dataset import make_city_sequences
from src.preprocessing import load_solar_csv
from src.train import build_model

def safe_mape(y_true,y_pred):
    mask=np.abs(y_true)>1.0
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100.0) if np.any(mask) else float("nan")

def main(config_path):
    with open(config_path,"r",encoding="utf-8") as f:
        cfg=yaml.safe_load(f)
    dc=cfg["data"]; out=Path(cfg["output"]["directory"]); model_type=cfg["model"]["type"].lower()
    ckpt=torch.load(out/f"{model_type}_best.pt",map_location="cpu")
    scaler=joblib.load(out/"scaler.joblib")

    frame=load_solar_csv(dc["csv_path"],dc["datetime_column"])
    test=frame[frame[dc["city_column"]].isin(dc["test_cities"])].copy()
    features=list(cfg["features"]["numeric"]); target=dc["target_column"]
    test=test.dropna(subset=features+[target]).copy()
    if test.empty:
        raise ValueError("No rows found for configured test_cities.")

    x,y,meta=make_city_sequences(test,features,target,dc["city_column"],
                                int(dc["sequence_length"]),scaler,dc["datetime_column"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(cfg,len(features),int(dc["sequence_length"]))
    model.load_state_dict(ckpt["model_state"]); model.to(device).eval()
    with torch.no_grad():
        pred=model(torch.as_tensor(x,dtype=torch.float32,device=device)).cpu().numpy()

    metrics={"MAE":float(mean_absolute_error(y,pred)),
             "RMSE":float(np.sqrt(mean_squared_error(y,pred))),
             "MAPE_percent_daylight":safe_mape(y,pred),
             "R2":float(r2_score(y,pred))}
    print(json.dumps(metrics,indent=2))
    with open(out/f"{model_type}_metrics.json","w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)

    pf=pd.DataFrame(meta); pf["actual_GHI"]=y; pf["predicted_GHI"]=pred
    pf.to_csv(out/f"{model_type}_predictions.csv",index=False)

    for city,g in pf.groupby("city"):
        g=g.copy(); g["datetime_utc"]=pd.to_datetime(g["datetime_utc"],utc=True)
        plt.figure(figsize=(10,4.8))
        plt.plot(g["datetime_utc"],g["actual_GHI"],label="Actual GHI")
        plt.plot(g["datetime_utc"],g["predicted_GHI"],label="Predicted GHI")
        plt.xlabel("Time"); plt.ylabel("GHI (W/m²)")
        plt.title(f"{city}: Actual vs Predicted GHI"); plt.legend(); plt.tight_layout()
        plt.savefig(out/f"{model_type}_{city}_ghi.png",dpi=180); plt.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/solar.yaml")
    main(p.parse_args().config)
