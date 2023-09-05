import pandas as pd

summary_path = "./logs/standard/Sampled_ImageNet_200x1000_50x100_Seed_6/summary_in_20230904.log"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


summary_df = pd.read_json(summary_path, lines=True)

select_df = summary_df[
    summary_df["model"].isin(["ContrastiveResNet18", "BasicQuantResNet18V3"])
][
    ['model', 'num_concepts', 'num_attended_concepts', 'norm_concepts', 'norm_summary', 'grad_factor',
     'loss_sparsity_weight', 'loss_sparsity_adaptive', 'loss_diversity_weight',
     'best_epoch', 'best_val_acc', 'best_val_acc_major', 'best_val_acc_subset_major',
     'best_val_acc_minor', 'best_val_acc_subset_minor', 'best_val_s50', 'best_val_s90',
     'best_val_loss_dvs', 'checkpoint_dir', 'detailed_log_path']
]


select_df[
    ['model', 'num_concepts', 'norm_concepts', 'norm_summary', 'grad_factor',
     'best_val_acc', 'best_val_acc_major', 'best_val_acc_subset_major',
     'best_val_acc_minor', 'best_val_acc_subset_minor', 'best_val_s50', 'best_val_s90',
     'best_val_loss_dvs']
].groupby(
    ['model', 'num_concepts', 'norm_concepts', 'norm_summary', 'grad_factor']
).mean().sort_values(
    "best_val_acc_subset_minor", ascending=False
).to_excel("ResNet18_Analysis.xlsx")
