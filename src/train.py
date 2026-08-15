from __future__ import annotations
import argparse, json, random
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch, yaml
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from src.dataset import SolarSequenceDataset, make_city_sequences
from src.models.lstm import LSTMRegressor
from src.models.transformer import TransformerRegressor
from src.preprocessing import load_solar_csv

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def build_model(cfg, input_size, sequence_length):
    mc = cfg["model"]
    common = dict(input_size=input_size, hidden_size=mc["hidden_size"],
                  num_layers=mc["num_layers"], dropout=mc["dropout"])
    if mc["type"].lower() == "lstm":
        return LSTMRegressor(**common)
    if mc["type"].lower() == "transformer":
        return TransformerRegressor(**common, num_heads=mc["num_heads"],
                                    max_length=max(512, sequence_length))
    raise ValueError("model.type must be 'lstm' or 'transformer'")

def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    dc = cfg["data"]; seq_len = int(dc["sequence_length"])
    frame = load_solar_csv(dc["csv_path"], dc["datetime_column"])
    features = list(cfg["features"]["numeric"])
    target, city_col, dt_col = dc["target_column"], dc["city_column"], dc["datetime_column"]

    required = set(features + [target, city_col, dt_col])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame = frame.dropna(subset=features + [target]).copy()
    test_cities = set(dc["test_cities"])
    trainval = frame[~frame[city_col].isin(test_cities)].copy()
    if trainval.empty:
        raise ValueError("No training rows remain after test city selection.")

    train_parts, val_parts = [], []
    vf = float(dc["validation_fraction"])
    for _, g in trainval.groupby(city_col, sort=False):
        g = g.sort_values(dt_col).reset_index(drop=True)
        if len(g) < 2*seq_len:
            continue
        cut = max(seq_len, int(len(g)*(1-vf)))
        cut = min(cut, len(g)-seq_len)
        train_parts.append(g.iloc[:cut].copy())
        val_parts.append(g.iloc[max(0, cut-seq_len+1):].copy())

    if not train_parts:
        raise ValueError("Training cities do not contain enough rows for the requested sequence length.")

    train_frame = pd.concat(train_parts, ignore_index=True)
    val_frame = pd.concat(val_parts, ignore_index=True)

    scaler = StandardScaler().fit(train_frame[features].to_numpy())
    xtr, ytr, _ = make_city_sequences(train_frame, features, target, city_col, seq_len, scaler, dt_col)
    xva, yva, _ = make_city_sequences(val_frame, features, target, city_col, seq_len, scaler, dt_col)

    tr_loader = DataLoader(SolarSequenceDataset(xtr,ytr), batch_size=cfg["training"]["batch_size"], shuffle=True)
    va_loader = DataLoader(SolarSequenceDataset(xva,yva), batch_size=cfg["training"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(features), seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])
    loss_fn = nn.MSELoss()

    out = Path(cfg["output"]["directory"]); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f'{cfg["model"]["type"].lower()}_best.pt'
    best, stale, history = float("inf"), 0, []

    for epoch in range(1, int(cfg["training"]["epochs"])+1):
        model.train(); tl=[]
        for x,y in tr_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(); pred=model(x); loss=loss_fn(pred,y); loss.backward(); opt.step()
            tl.append(loss.item())
        model.eval(); vl=[]
        with torch.no_grad():
            for x,y in va_loader:
                x,y=x.to(device),y.to(device)
                vl.append(loss_fn(model(x),y).item())
        tr_loss=float(np.mean(tl)); va_loss=float(np.mean(vl))
        history.append({"epoch":epoch,"train_mse":tr_loss,"val_mse":va_loss})
        print(f"Epoch {epoch:03d} | train={tr_loss:.4f} | val={va_loss:.4f}")
        if va_loss < best:
            best, stale = va_loss, 0
            torch.save({"model_state":model.state_dict(),"features":features,"config":cfg}, checkpoint)
        else:
            stale += 1
            if stale >= int(cfg["training"]["patience"]):
                print("Early stopping."); break

    joblib.dump(scaler, out/"scaler.joblib")
    with open(out/"training_history.json","w",encoding="utf-8") as f:
        json.dump(history,f,indent=2)
    print(f"Saved: {checkpoint}")

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/solar.yaml")
    main(p.parse_args().config)
