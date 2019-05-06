# Issue #4: understand data, EDA

### 1.1. Profiles DF

This dataframe is a one-hot encoding of the profiles of size `(63090, 67)`. Each profile has an encoding of an array of length 67. There are 63090 profiles in total. No null values

### 1.2. Queries DF

This dataframe shows the 500.000 queries made during the timeframe from October 1 at 00:00:10 until November 30 at 23:59:17. 

* Queries per day
    - The date is in this format: `2018-10-20 17:08:12`
    - Preprocessing: obtain the dates
    - Increment by 30 every month
    - With the input divided across 2 months, the dates range between 1 and 60.
    - Visualize count of queries per day

![ ](figures/queries_day.png)

The amount of queries per day is quite consistent #TODO average per day and check weekends?

Statistics of `df_queries`

![ ](figures/df_queries_stats.png){width=400}

* 500k entries for all columns except for `pid`, there are 164k anonymous entries. The rest are done by 46k users (one user can make many queries).
* The average day is 29, approximately the middle of the array of time, at the end of the first month. The queries are more or less constant throughout the days.
* The coordinates are heavily centered in (116,399). They vary from (115-117) and (39.4,40.9)

* User analysis
    - There are 46k users
    - Each user has made an average of 7 queries
    - There's a high variance (54)
    - The most popular users have 6000 entries, while many more have just one
    - Almost 27k users have done only one or two queries.

These are the number of queries of the most popular users

![ ](figures/queries_user.png){width=400}

### 1.3. Clicks DF

No null values. There are 453k rows, but 500k queries. TODO: why less?

![ ](figures/clicks_per_mode.png){width=400}

The classes are quite imbalanced, almost 140k entries for mode=2 and just 5k for mode=8. 

* Possible approaches:
    - Downsample mode=2 and probably 1 and 7
    - Upsample 4, 6, 8, 11? Dangerous

The F1-score should take into account all classes but there are 30x times more entries for mode=2 than for mode=8, mode=8 won't be classified that well.

### 1.4. Plans DF

No null values. There are 491k entries, but 500k queries. TODO: why less? Some queries don't have plans?

TODO: how to process plans? new rows, new columns?

