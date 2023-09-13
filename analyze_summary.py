import pandas as pd

summary_path = "./logs/standard/Sampled_ImageNet_200x1000_200x25_Seed_6/summary_of_V4Smooth.log"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


summary_df = pd.read_json(summary_path, lines=True)

select_df = summary_df[
    summary_df["model"].isin(
        ["BasicQuantResNet18V4", "BasicQuantResNet18V4Smooth"]
    )
][
    ['model', 'num_concepts', 'num_attended_concepts', 'norm_concepts',
     'loss_sparsity_weight', 'loss_sparsity_adaptive', 'loss_diversity_weight',
     'best_epoch', 'best_train_acc', 'best_val_acc', 'best_val_acc_major', 'best_val_acc_subset_major',
     'best_val_acc_minor', 'best_val_acc_subset_minor', 'best_val_s50', 'best_val_s90', 'best_val_loss_dvs',
     'checkpoint_dir', 'detailed_log_path']
]


grouped_df = select_df[
    ['model', 'num_concepts', 'num_attended_concepts', 'norm_concepts', 'loss_sparsity_weight', 'loss_diversity_weight',
     'best_epoch', 'best_train_acc', 'best_val_acc', 'best_val_acc_major', 'best_val_acc_subset_major',
     'best_val_acc_minor', 'best_val_acc_subset_minor', 'best_val_s50', 'best_val_s90', 'best_val_loss_dvs']
].groupby(
    ['loss_sparsity_weight', 'model', 'num_concepts',
        'num_attended_concepts', 'norm_concepts', 'loss_diversity_weight']
)

pd.concat(
    [
        grouped_df.mean(),
        grouped_df.size().to_frame("count")
    ],
    axis=1
).reset_index()
