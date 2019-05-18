def optical_flow_dense(image_current, image_next):
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next    = cv2.cvtColor(image_next,    cv2.COLOR_RGB2GRAY)
    hsv          = np.zeros((66, 220, 3))
    hsv[:, :, 1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:, :, 1]
    
    flow_mat   = None
    pyr_scale  = 0.5
    levels     = 1
    winsize    = 15
    iterations = 2
    poly_n     = 5
    poly_sigma = 1.1
    flags      = 0
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next, flow_mat, pyr_scale,
                                        levels, winsize, iterations, poly_n, poly_sigma,
                                        flags)
    # Convert from Catesian to Polar coordinates:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[:, :, 0]     = angle * (180 / np.pi / 2)
    hsv[:, :, 2]     = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv              = np.asarray(hsv, dtype=np.float32)
    rgb_flow         = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb_flow

