import pandas as pd

summary_path = "./logs/standard/summary.log"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


summary_df = pd.read_json(summary_path, lines=True)

select_df = summary_df[
    summary_df["use_model"].isin(
        ["OriTextCQPoolResNet18"]
    )
]
# select_df[["concept_attn_head", "patch_attn_head"]] = select_df[[
#     "concept_attn_head", "patch_attn_head"]].fillna("")
# select_df[["concept_attn_max_fn", "patch_attn_max_fn"]] = select_df[[
#     "concept_attn_max_fn", "patch_attn_max_fn"]].fillna("")
select_df["expand_dim"] = select_df["expand_dim"].fillna(False)
select_df["expand_dim"] = select_df["expand_dim"] != 0

grouped_df = select_df[
    ['use_model', 'num_concepts', 'num_attended_concepts', 'norm_concepts', 'att_smoothing',
     'concept_attn_head', 'concept_attn_max_fn', 'patch_attn_head', 'patch_attn_max_fn', 'expand_dim',
     'loss_sparsity_weight', 'loss_sparsity_adaptive', 'loss_diversity_weight',
     'best_epoch', 'best_train_acc', 'best_val_acc', 'best_major_acc', 'best_major_acc_subset', 'best_minor_acc', 'best_minor_acc_subset',
     'best_train_cfi_s10', 'best_train_cfi_s50', 'best_train_cfi_s90',
     'best_train_pfi_s10', 'best_train_pfi_s50', 'best_train_pfi_s90',
     'best_train_cfp_s10', 'best_train_cfp_s50', 'best_train_cfp_s90',
     'best_minor_cfi_s10', 'best_minor_cfi_s50', 'best_minor_cfi_s90',
     'best_minor_pfi_s10', 'best_minor_pfi_s50', 'best_minor_pfi_s90',
     'best_minor_cfp_s10', 'best_minor_cfp_s50', 'best_minor_cfp_s90',
     'best_train_loss_dvs', 'num_classes']
].groupby(
    ['num_classes', 'use_model', 'att_smoothing', 'num_concepts', 'loss_sparsity_weight', 'num_attended_concepts',
     'concept_attn_head', 'concept_attn_max_fn', 'patch_attn_head', 'patch_attn_max_fn', 'expand_dim',
     'loss_sparsity_adaptive', 'norm_concepts', 'loss_diversity_weight']
)

pd.concat(
    [
        grouped_df.mean(),
        grouped_df.size().to_frame("count")
    ],
    axis=1
).reset_index()
