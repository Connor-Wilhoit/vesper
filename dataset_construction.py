def dataset_construction(video_location, image_folder, total_frames, dataset_type):
    meta_dict = {}
    
    tqdm.write('reading in video file....')
    total_frames = total_train_frames
    cap          = cv2.VideoCapture(video_location)
    count        = 0
    tqdm.write('constructing dataset....')
    
    while (True):
        # Extract individual frame
        idx, frame = cap.read()
        if not idx:
            break
        
        img_path    = os.path.join(image_folder, str(count)+'.jpg')
        #frame_speed = train_y_list[count]
        frame_speed = float('NaN') if dataset_type == 'test' else train_y_list[idx]
        
        meta_dict[count] = [img_path, count, frame_speed]
        cv2.imwrite(img_path, frame)
        count += 1
        
        if (count % 1000)==0:
            tqdm.write("{} frames processed so far".format(count))
        if dataset_type == 'train' and count == 20400:
            tqdm.write("""all frames should be done being processed.
            current frame # ---> {}""".format(count))
        if dataset_type == 'test' and count == 10978:
            tqdm.write("""all frames should be done being processed.
            current frame # ---> {}""".format(count))
            
            
    meta_df         = pd.DataFrame.from_dict(meta_dict, orient='index')
    meta_df.columns = ['image_path', 'image_index', 'speed']
    
    tqdm.write('writing meta_df to csv')
    meta_df.to_csv(os.path.join(clean_data_path, dataset_type+'_meta.csv'), index=False)

    return "done ---> dataset_construction"
