def generate_training_data(data, batch_size=32):
    image_batch = np.zeros((batch_size, 66, 220, 3))    # vesper's input parameters
    label_batch = np.zeros((batch_size))
    
    while True:
        for i in range(batch_size):
            # generate a random index w/a uniform random distribution form 1 to len - 1
            idx = np.random.randint(1, len(data) - 1)
            
            # generate a random brightness_factor to apply to both images:
            brightness_factor = 0.2 + np.random.uniform()
            
            row_now  = data.iloc[[idx]].reset_index()
            row_prev = data.iloc[[idx - 1]].reset_index()
            row_next = data.iloc[[idx + 1]].reset_index()
            
            # find the 3 respective times todetermine frame order (current --> next)
            time_now  = row_now['image_index'].values[0]
            time_prev = row_prev['image_index'].values[0]
            time_next = row_next['image_index'].values[0]
            
            if abs(time_now - time_prev) == 1 and time_now > time_prev:
                row1 = row_prev
                row2 = row_now
                
            elif abs(time_next - time_now) == 1 and time_next > time_now:
                row1 = row_now
                row2 = row_next
                
            else:
                print("Error generating row")
                
            # process first image
            x1, y1 = preprocess_image_from_path(row1['image_path'].values[0], row1['speed'].values[0], brightness_factor)
            
            # process second image
            x2, y2 = preprocess_image_from_path(row2['image_path'].values[0], row2['speed'].values[0], brightness_factor)
            
            # compute optical flow (send in images as RGB)
            rgb_diff = optical_flow_dense(x1, x2)
            
            # calculate mean speed (average speed) [slope of the secant line]
            y              = np.mean([y1, y2])            
            image_batch[i] = rgb_diff
            label_batch[i] = y
            
            # shuffle the pairs before the get fed in as input to our Neural Network
            yield shuffle(image_batch, label_batch)

