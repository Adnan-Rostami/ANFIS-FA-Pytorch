
import numpy as np, time, pandas as pd, os
from copy import deepcopy
from evalfis import evalfis
from metrics_eval import metrics_eval
from TrainFisCost import TrainFISCost

def TrainUsingFA(fis_init, TrainData, Params, save_path=None):
    
    MaxIt = Params.get("MaxIt", 50)
    nPop  = Params.get("nPop", 50)
    γ, β0, α, α_damp = 1.0, 2.0, 0.2, 0.98

    # جمعیت اولیه
    dim = len(np.random.uniform(-1, 1, 10))  # یا تعداد پارامترهای واقعی FIS
    Pop = [{
        "Position": np.random.uniform(-1, 1, dim),
        "Cost": None,
        "fis": deepcopy(fis_init)
    } for _ in range(nPop)]

    # هزینه اولیه
    Cost = np.zeros(nPop)
    for i in range(nPop):
        rmse, out = TrainFISCost(Pop[i]["Position"], {**TrainData, "fis": deepcopy(fis_init)})
        Cost[i] = rmse
        Pop[i]["fis"] = out["fis"]

    BestCost, records = [], []
    t0 = time.time()

    for it in range(MaxIt):
        # حرکت کرم‌ها
        NewPop, NewCost = deepcopy(Pop), deepcopy(Cost)
        for i in range(nPop):
            for j in range(nPop):
                if Cost[j] < Cost[i]:
                    r = 1.0
                    β = β0 * np.exp(-γ * r**2)
                    # حرکت پارامترها
                    for k in range(len(Pop[i]['fis']['rules'])):
                        rule_i, rule_j = Pop[i]['fis']['rules'][k], Pop[j]['fis']['rules'][k]
                        means_i, means_j = np.array(rule_i['means']), np.array(rule_j['means'])
                        sigmas_i, sigmas_j = np.array(rule_i['sigmas']), np.array(rule_j['sigmas'])
                        cons_i, cons_j   = np.array(rule_i['consequents']), np.array(rule_j['consequents'])
                        NewPop[i]['fis']['rules'][k]['means']       = (means_i + β*(means_j-means_i) + α*(np.random.rand(len(means_i))-0.5)).tolist()
                        NewPop[i]['fis']['rules'][k]['sigmas']      = np.clip(sigmas_i + β*(sigmas_j-sigmas_i) + α*(np.random.rand(len(sigmas_i))-0.5), 1e-6, 10).tolist()
                        NewPop[i]['fis']['rules'][k]['consequents'] = (cons_i + β*(cons_j-cons_i) + α*(np.random.rand(len(cons_i))-0.5)).tolist()
            rmse, out = TrainFISCost(NewPop[i]["Position"], {**TrainData, "fis": deepcopy(NewPop[i]['fis'])})
            NewCost[i] = rmse
            NewPop[i]["fis"] = out["fis"]

        Pop, Cost = deepcopy(NewPop), deepcopy(NewCost)
        α *= α_damp
        BestCost.append(np.min(Cost))
        best_idx = np.argmin(Cost)
        fis_best = Pop[best_idx]["fis"]

        # --- متریک آموزش
        y_train_prob = evalfis(TrainData["Inputs"], fis_best).flatten()
        metrics_train, _ = metrics_eval(TrainData["Targets"], y_train_prob)



        if save_path:
            os.makedirs(save_path, exist_ok=True)
            csv_path = os.path.join(save_path, "metrics_history.csv")
            df_iter = pd.DataFrame([records[-1]])  # فقط رکورد آخر
            header_needed = not os.path.exists(csv_path)
            df_iter.to_csv(csv_path, mode='a', index=False, header=header_needed)
            
 


        print( f"Iter {it+1}"   ,records)

    final_best_fis = Pop[np.argmin(Cost)]["fis"]
    return {
        "bestfis": final_best_fis,
        "BestCost": BestCost[-1],
        "BestHistory": BestCost,
        "MetricsHistory": pd.DataFrame(records)
    }
