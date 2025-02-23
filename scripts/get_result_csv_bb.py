import pandas as pd
from statistics import mean

csv_path1 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_no_margin_no_entropy_hp__n3849_top-4_lowrank-False.csv"
csv_path2 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_error-inj_n3849_top-4_lowrank-False.csv"
csv_path3 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_error-inj_hp__n3849_top-4_lowrank-False.csv"
csv_path4 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_entropy_layers_hp_[27, 32]_n3849_top-4_lowrank-False.csv"
csv_path5 ="/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_margin_layers_hp_[22, 32]_n3849_top-4_lowrank-False.csv"
csv_path6 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_dummy_hp_n3849_top-4_lowrank-True.csv"


csv_path1 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_no_margin_no_entropy_pd__n3849_top-4_lowrank-False.csv"
csv_path2 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_error-inj_n3849_top-4_lowrank-False.csv"
csv_path3 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_error-inj_pd__n3849_top-4_lowrank-False.csv"

csv_path4 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_entropy_layers_pd_[27, 32]_n3849_top-4_lowrank-False.csv"
csv_path5 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_margin_layers_pd_[22, 32]_n3849_top-4_lowrank-False.csv"
csv_path6 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_ROME_outputs_zsre_editing_sweep_ws-[1]_layer-7_dummy_pd_n3849_top-4_lowrank-True.csv"

csv_path1 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_bb_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse.csv"
csv_path2 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_jailbreak_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse.csv"
csv_path3 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_mg_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse.csv"

csv_path4 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_entropy_layers_bb_img_attack_parap_orig[22, 32]_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elTrue.csv"
csv_path5 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_entropy_layers_mg_img_attack_parap_orig[22, 32]_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elTrue.csv"
csv_path6 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_entropy_layers_jailbreak_img_attack_parap_orig[22, 32]_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elTrue.csv"

csv_path7 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_margin_layers_bb_img_attack_parap_orig[22, 32]_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlTrue_elFalse.csv"
csv_path8 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_margin_layers_mg_img_attack_parap_orig[22, 32]_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlTrue_elFalse.csv"
csv_path9 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_margin_layers_jailbreak_img_attack_parap_orig[22, 32]_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlTrue_elFalse.csv"

csv_path10 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_erro_inj_jailbreak_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse.csv"
csv_path11 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_erro_inj_bb_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse.csv"
csv_path12 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_erro_inj_mg_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse.csv"

csv_path13 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_dummy_mg_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse.csv"
csv_path14 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_dummy_bb_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse.csv"
csv_path15 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_dummy_jailbreak_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse.csv"


#Input_rephrase
#lang=attack
csv_path16 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_bb_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse_faeFalse.csv"
csv_path17 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_jailbreak_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse_faeFalse.csv"
csv_path18 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_mg_img_attack_parap_orig_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse_faeFalse.csv"

#img_tatcl
csv_path19 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_img_img_attack_parap_medium_only_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse_faeFalse.csv"
csv_path20 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_img_img_attack_parap_easy_only_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse_faeFalse.csv"
csv_path21 = "/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_img_img_attack_parap_hard_only_n600_top-4_lowrank-False_parap_image_lfteditTrue_cfteditFalse_mlFalse_elFalse_faeFalse.csv"


paths = [globals()["csv_path{}".format(k)] for k in range(1,22,1)]#, csv_path6]


# paths = ["/nas-ssd2/vaidehi/MMMEdit/belief-localization/results/llava-v1.5-7b_FT_outputs_zsre_editing_sweep_ws-[1]_layer-7_fact-erasure_no_margin_no_entropy_hp__n500_top-4_lowrank-False.csv"]

# metrics = ['topk_metric_agg', 'bottomk_metric_agg',
#        'topk_or_bottomk_metric_agg', 'top1_metric_agg', 'bottom1_metric_agg',                                                               
#        'top1_or_bottom1_metric_agg', 'delta_accuracy',
#        'delta_accuracy_neighborhood','neighborhood_score', 'rewrite_score', 'paraphrase_score']

# metrics = ['topk_metric_agg', 'bottomk_metric_agg',
#        'topk_or_bottomk_metric_agg', 'top1_metric_agg', 'bottom1_metric_agg',                                                               
#        'top1_or_bottom1_metric_agg', 'target_prob_agg', 
#        'topk_pre_metric_agg', 'bottomk_pre_metric_agg','topk_or_bottomk_pre_metric_agg', 'top1_pre_metric_agg',
#        'bottom1_pre_metric_agg', 'top1_or_bottom1_pre_metric_agg',
#        'target_prob_pre_agg','retain_rate', 'retain_rate_neighborhood',
#        'retain_rate_pre', 'retain_rate_neighborhood_pre', 'delta_accuracy',
#        'delta_accuracy_neighborhood','neighborhood_score', 'rewrite_score', 'paraphrase_score']

metrics = [ 'tgt_in_sample', 'attack_frac', 'tgt_in_sample_pre', 'attack_frac_pre',
       'retain_rate', 'retain_rate_neighborhood', 'retain_rate_pre',
       'retain_rate_neighborhood_pre', 'delta_accuracy', 'delta_accuracy_neighborhood', 'rewrite_score', 'neighborhood_score', 'paraphrase_score']

result = []
df = pd.DataFrame()

for path in paths:
    x = pd.read_csv(path)
    metrics_cur = []
    x = x[metrics]
    metrics_cur.append(path.split("/")[-1])
    for metric in metrics:
        metrics_cur.append(mean(x[metric]))
    result.append(metrics_cur)

df = pd.DataFrame(result, columns=['Name']+metrics)

print(df.head())

df.to_csv("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/result_agg.csv")

# print(x.columns)
# result["name"] = csv_path.split("/")[-1]
# for metric in metrics:
#     print(metric)
#     # print(x[metric])
#     result[metric] = mean(x[metric])




pd.Series(result).to_csv("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/result.csv")


