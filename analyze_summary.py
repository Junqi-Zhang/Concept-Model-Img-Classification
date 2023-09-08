import pandas as pd

summary_path = "./logs/standard/Sampled_ImageNet_200x1000_200x25_Seed_6/summary_in_ResNet18.log"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


summary_df = pd.read_json(summary_path, lines=True)

select_df = summary_df[
    summary_df["model"].isin(
        ["ResNet18", "BasicQuantResNet18V4", "ContrastiveResNet18"]
    )
][
    ['model', 'num_concepts', 'num_attended_concepts',
     'loss_sparsity_weight', 'loss_sparsity_adaptive', 'loss_diversity_weight', 'batch_size',
     'best_epoch', 'best_train_acc', 'best_val_acc', 'best_val_acc_major', 'best_val_acc_subset_major',
     'best_val_acc_minor', 'best_val_acc_subset_minor', 'best_val_s50', 'best_val_s90',
     'checkpoint_dir', 'detailed_log_path']
]


select_df[
    ['model', 'num_concepts', 'num_attended_concepts', 'loss_sparsity_adaptive',
     'best_train_acc', 'best_val_acc', 'best_val_acc_major', 'best_val_acc_subset_major',
     'best_val_acc_minor', 'best_val_acc_subset_minor', 'best_val_s50', 'best_val_s90']
].groupby(
    ['model', 'num_concepts', 'num_attended_concepts',
     'loss_sparsity_adaptive']
).mean().sort_values(
    "best_val_acc_subset_minor", ascending=False
).to_excel("ResNet18.xlsx")
