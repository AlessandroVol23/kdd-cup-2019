import pandas as pd

''' 
# Handling imbalanced data sets by sampling

## Down-sampling = Randomly removing observations from a class to prevent its signal from dominating the learning algorithm
## Up-sampling = Randomly duplicating observations from a class in order to reinforce its signal


--- Following functions are used to sample click_modes ---

# sampling_percentage(mode, percentage, df)
    
    mode        = click_mode to be sampled
    percentage  = % of chosen click_mode, i.e. up-sample percentage > 100%; down-sample percentage < 100%
                    e.g. putting in 30 for percentage will remove 70% of chosen click_mode 
    df          = source data frame

# sampling_percentage(mode, percentage, df)
    
    mode        = click_mode to be sampled
    amount      = desired amount to which chosen mode should be sampled up/down to
    df          = source data frame


Both functions will return df_final (new data frame including sampled click_mode)    
'''

def sampling_percentage(mode, percentage, df):
    df_rest_modes = df.loc[df.click_mode != mode]

    df_sample_mode = df.loc[df.click_mode == mode]
    quantity = int(percentage * 0.01 * len(df_sample_mode))+1

    df_sample_mode_target = df_sample_mode.sample(quantity, replace=True)

    df_final = pd.concat([df_sample_mode_target, df_rest_modes], axis=0)

    return df_final


def sampling_amount(mode, amount, df):
    df_rest_modes = df.loc[df.click_mode != mode]
    df_sample_mode = df.loc[df.click_mode == mode]

    df_sample_mode_target = df_sample_mode.sample(amount, replace=True)

    df_final = pd.concat([df_sample_mode_target, df_rest_modes], axis=0)

    return df_final