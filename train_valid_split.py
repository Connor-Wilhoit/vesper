def train_valid_split(dataframe, seed_value):
    # Randomly shuffle pairs of rows in the dataframe, separates training and validation data,
    # generates a uniform random variable 0->9, gives 20% chance to append to valid_data, otherwise train_data
    # return tupel (train_data, valid_data) dataframes.
    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()
    np.random.seed(seed_value)
    for i in tqdm(range(len(dataframe) - 1)):
        idx1 = np.random.randint(len(dataframe) - 1)
        idx2 = idx1 + 1
        
        row1 = dataframe.iloc[[idx1]].reset_index()
        row2 = dataframe.iloc[[idx2]].reset_index()
        
        random_int = np.random.randint(9)
        if 0 <= random_int <= 1:
            valid_frames = [valid_data, row1, row2]
            valid_data   = pd.concat(valid_frames, axis=0, join='outer', ignore_index=False)

        if random_int >= 2:
            train_frames = [train_data, row1, row2]
            train_data   = pd.concat(train_frames, axis=0, join='outer', ignore_index=False)
            
    return train_data, valid_data

