import pandas as pd

summary_path = "./logs/standard/Sampled_ImageNet/summary_in_20230829.log"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

summary_df = pd.read_json(summary_path, lines=True)
