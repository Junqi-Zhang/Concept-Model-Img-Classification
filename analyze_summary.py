import pandas as pd

summary_path = "logs/summary_OriTextHierarchicalConceptualPoolResNet.log"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


summary_df = pd.read_json(summary_path, lines=True)

select_df = summary_df[
    summary_df["use_model"].isin(
        ["OriTextHierarchicalConceptualPoolResNet"]
    )
]

grouped_df = select_df[
    ['supplementary_description', 'use_model', 'backbone_name',
     'num_low_concepts', 'num_attended_low_concepts', 'num_high_concepts', 'num_attended_high_concepts',
     'low_high_max_function', 'output_high_concepts_type', 'detach_low_concepts',
     'patch_low_concept_num_heads', 'patch_low_concept_max_function', 'image_patch_num_heads', 'image_patch_max_function',
     'loss_low_diversity_weight',
     'loss_high_sparsity_weight', 'loss_high_diversity_weight', 'loss_aux_classification_weight',
     'best_epoch', 'best_train_A', 'best_train_A_aux', 'best_major_A_sub', 'best_major_A_auxsub',
     'best_minor_A', 'best_minor_A_sub', 'best_minor_A_aux', 'best_minor_A_auxsub',
     'best_train_pfi_s10', 'best_train_pfi_s50', 'best_train_pfi_s90',
     'best_train_lcfi_s10', 'best_train_lcfi_s50', 'best_train_lcfi_s90',
     'best_train_hcfi_s10', 'best_train_hcfi_s50', 'best_train_hcfi_s90',
     'best_train_lcfp_s10', 'best_train_lcfp_s50', 'best_train_lcfp_s90',
     'best_train_hcfp_s10', 'best_train_hcfp_s50', 'best_train_hcfp_s90',
     'best_train_lfh_s10', 'best_train_lfh_s50', 'best_train_lfh_s90',
     'best_major_pfi_s10', 'best_major_pfi_s50', 'best_major_pfi_s90',
     'best_major_lcfi_s10', 'best_major_lcfi_s50', 'best_major_lcfi_s90',
     'best_major_hcfi_s10', 'best_major_hcfi_s50', 'best_major_hcfi_s90',
     'best_major_lcfp_s10', 'best_major_lcfp_s50', 'best_major_lcfp_s90',
     'best_major_hcfp_s10', 'best_major_hcfp_s50', 'best_major_hcfp_s90',
     'best_major_lfh_s10', 'best_major_lfh_s50', 'best_major_lfh_s90',
     'best_minor_pfi_s10', 'best_minor_pfi_s50', 'best_minor_pfi_s90',
     'best_minor_lcfi_s10', 'best_minor_lcfi_s50', 'best_minor_lcfi_s90',
     'best_minor_hcfi_s10', 'best_minor_hcfi_s50', 'best_minor_hcfi_s90',
     'best_minor_lcfp_s10', 'best_minor_lcfp_s50', 'best_minor_lcfp_s90',
     'best_minor_hcfp_s10', 'best_minor_hcfp_s50', 'best_minor_hcfp_s90',
     'best_minor_lfh_s10', 'best_minor_lfh_s50', 'best_minor_lfh_s90',
     'best_val_L_ldvs', 'best_val_L_hdvs',
     'best_val_L_lsps', 'best_val_L_hsps']
].groupby(
    ['supplementary_description', 'use_model', 'backbone_name',
     'num_low_concepts', 'num_attended_low_concepts', 'num_high_concepts', 'num_attended_high_concepts',
     'low_high_max_function', 'output_high_concepts_type', 'detach_low_concepts',
     'patch_low_concept_num_heads', 'patch_low_concept_max_function', 'image_patch_num_heads', 'image_patch_max_function',
     'loss_low_diversity_weight',
     'loss_high_sparsity_weight', 'loss_high_diversity_weight', 'loss_aux_classification_weight',]
)

pd.concat(
    [
        grouped_df.mean(),
        grouped_df.size().to_frame("count")
    ],
    axis=1
).reset_index()
