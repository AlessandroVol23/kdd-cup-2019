import pandas as pd
from clickmodes_by_districts import get_click_modes, get_districts, plot_click_modes_by_districts


df_districts = pd.read_pickle("../data/external/districts/train_all_first_districts.pickle")

click_modes = get_click_modes(df_districts)
districts = get_districts(df_districts)
plot_click_modes_by_districts(df_districts, click_modes, districts, 'origin')
print('Plotting successful')
