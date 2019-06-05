

import pandas as pd
pd.set_option('display.max_rows', 500)

res = pd.read_json('results_sepa_linear.json')

r = res.filter(items=['train_config.supervised_vae.omega', 'train_config.supervised_vae.beta', 'train_config.dataset.name', 'evaluation_config.dataset.name', 'evaluation_results.eval_accuracy'])
rg = r.groupby(['train_config.supervised_vae.omega', 'train_config.supervised_vae.beta','train_config.dataset.name', 'evaluation_config.dataset.name']).agg({'evaluation_results.eval_accuracy': ['mean', 'std', 'count']})
print(rg)
