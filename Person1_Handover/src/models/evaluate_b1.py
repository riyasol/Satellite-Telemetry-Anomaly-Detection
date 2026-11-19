import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.utils.config import load_config, resolve_path
from src.models.anomaly_scoring import (
    load_forecaster, score_windows, ema_reset_on_excluded, window_has_anomaly
)

def basic_metrics(y_true, y_pred):
    y_true=np.asarray(y_true,dtype=bool); y_pred=np.asarray(y_pred,dtype=bool)
    tp=int(np.sum(y_true & y_pred)); fp=int(np.sum(~y_true & y_pred))
    fn=int(np.sum(y_true & ~y_pred)); tn=int(np.sum(~y_true & ~y_pred))
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    beta2=0.25
    f05=(1+beta2)*precision*recall/(beta2*precision+recall) if precision+recall else 0.0
    far=fp/(fp+tn) if fp+tn else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":precision,
            "recall":recall,"f0.5_point_window":f05,"false_alarm_rate":far}

def operational_test_ema(scores, alpha=0.2):
    """Causal operational EMA used after the validation threshold is frozen.

    This function is deliberately independent of exact ground-truth labels.
    P1-P2-P3 online decisions must use it or an identical configured EMA
    implementation without feeding test labels into the stream.
    """
    scores = np.asarray(scores, dtype=float)
    if not len(scores):
        return scores.copy()
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("EMA alpha must be in [0, 1]")
    out = np.empty(len(scores), dtype=float)
    state = None
    for i, score in enumerate(scores):
        if not np.isfinite(score):
            state = None
            out[i] = np.nan
            continue
        state = float(score) if state is None else float(alpha) * float(score) + (1.0 - float(alpha)) * state
        out[i] = state
    return out

def main():
    cfg=load_config(); m=cfg["model"]; a=cfg["anomaly"]
    device_name=cfg.get("device","auto")
    device="cuda" if device_name=="auto" and torch.cuda.is_available() else ("cpu" if device_name=="auto" else device_name)
    stride=int(m.get("window_stride",10))
    for mission in cfg["data"].get("missions",["Mission1"]):
        pdir=resolve_path(cfg["data"]["esa_work_dir"])/mission.lower()
        data=np.load(pdir/"splits.npz",allow_pickle=True)
        values,labels,exclusion,timestamps=data["values"],data["labels"],data["evaluation_ignore_mask"],data["timestamps_ns"]
        vs,ve=data["validation_bounds"].tolist(); ts,te=data["test_bounds"].tolist()
        model_dir=resolve_path(cfg["output"]["model_dir"])/mission.lower()
        model=load_forecaster(model_dir/"lstm_b1.pt",device)

        vi,vscores,_,_=score_windows(model,values,vs,ve,m["input_window"],m["forecast_horizon"],
            m["batch_size"],device,stride)
        vbad=window_has_anomaly(data["exclusion_mask"],vi,m["input_window"],m["forecast_horizon"],include_input=True)
        vema=ema_reset_on_excluded(vscores,~vbad,a["ema_alpha"])
        normal_scores=vema[np.isfinite(vema)]
        if not len(normal_scores):
            raise RuntimeError(f"{mission}: no eligible normal validation windows for threshold.")
        threshold=float(np.quantile(normal_scores,a["threshold_quantile"]))

        ti,tscores,preds,targets=score_windows(model,values,ts,te,m["input_window"],
            m["forecast_horizon"],m["batch_size"],device,stride)
        # Ignore windows whose target horizon intersects an annotation crossing a split boundary.
        test_ignore=window_has_anomaly(exclusion,ti,m["input_window"],m["forecast_horizon"],include_input=False)
        tema=ema_reset_on_excluded(tscores,~test_ignore,a["ema_alpha"])
        # Exact ground truth only; expanded margins never alter evaluation labels.
        true=window_has_anomaly(labels,ti,m["input_window"],m["forecast_horizon"],include_input=False)
        eligible=np.isfinite(tema)
        predicted=(tema>threshold) & eligible
        metrics=basic_metrics(true[eligible],predicted[eligible])
        metrics.update({"mission":mission,"threshold":threshold,
            "threshold_quantile":float(a["threshold_quantile"]),"ema_alpha":float(a["ema_alpha"]),
            "validation_windows":int(len(vi)),"normal_validation_windows":int(np.isfinite(vema).sum()),
            "test_windows":int(len(ti)),"test_windows_excluded_for_boundary_crossing":int(test_ignore.sum()),"test_windows_scored":int(eligible.sum()),"window_stride":stride,
            "score_definition":"MSE over forecast horizon and selected channels, followed by EMA across test windows.",
            "ground_truth_definition":"Exact annotation intervals only; configured margins are used only for exclusion.",
            "caveat":"Point-window metrics are not event-wise F0.5; event matching must be agreed with the team. Boundary-crossing annotation windows are excluded from scoring."})
        outdir=resolve_path(cfg["output"]["results_dir"])/mission.lower()
        plotdir=resolve_path(cfg["output"]["plot_dir"])/mission.lower()
        outdir.mkdir(parents=True,exist_ok=True); plotdir.mkdir(parents=True,exist_ok=True)
        (outdir/"b1_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        score_times=timestamps[ti+m["input_window"]]
        pd.DataFrame({"forecast_start_index":ti,"forecast_start_utc":pd.to_datetime(score_times,utc=True),
            "raw_mse":tscores,"ema_score":tema,"threshold":threshold,
            "predicted_anomaly":predicted.astype(np.int8),"true_target_window_anomaly":true.astype(np.int8)
            ,"valid":np.isfinite(tema).astype(np.int8)
            ,"eligible_for_drift":(~test_ignore & np.isfinite(tema)).astype(np.int8)
            ,"mask_reason":np.where(
                test_ignore,
                "expanded_annotation_or_boundary_exclusion",
                np.where(np.isfinite(tema), "eligible", "invalid_score")
            )
        }).to_csv(outdir/"b1_scores.csv",index=False)

        plt.figure(figsize=(13,4)); plt.plot(pd.to_datetime(score_times,utc=True),tema,label="EMA residual score")
        plt.axhline(threshold,linestyle="--",label="Normal-validation threshold")
        plt.xlabel("UTC time"); plt.ylabel("EMA MSE"); plt.legend(); plt.tight_layout()
        plt.savefig(plotdir/"b1_residual_scores.png",dpi=150); plt.close()
        if len(preds):
            sample=min(10,len(preds)-1)
            plt.figure(figsize=(10,4))
            plt.plot(targets[sample,:,0],marker="o",label="Actual")
            plt.plot(preds[sample,:,0],marker="x",label="Forecast")
            plt.xlabel("Forecast step"); plt.ylabel("Normalized value")
            plt.title(f"{mission}: example forecast, {data['channel_names'][0]}")
            plt.legend(); plt.tight_layout()
            plt.savefig(plotdir/"b1_actual_vs_predicted.png",dpi=150); plt.close()
        print(f"\n{mission} B1 metrics:\n{json.dumps(metrics,indent=2)}")
        print(f"Saved to {outdir}")

if __name__=="__main__":
    main()
